from tqdm import tqdm
import pandas as pd
import torch

from attention_dynamic_model import set_decode_type
from reinforce_baseline import validate

from utils import generate_data_onfly, get_results, get_cur_time
from time import gmtime, strftime


def train_model(optimizer,
                model_torch,
                baseline,
                validation_dataset,
                samples = 1280000,
                batch = 128,
                val_batch_size = 1000,
                start_epoch = 0,
                end_epoch = 5,
                from_checkpoint = False,
                grad_norm_clipping = 1.0,
                batch_verbose = 1000,
                graph_size = 20,
                filename = None
                ):

    if filename is None:
        filename = 'VRP_{}_{}'.format(graph_size, strftime("%Y-%m-%d", gmtime()))

    def rein_loss(model, inputs, baseline, num_batch):
        """Calculate loss for REINFORCE algorithm
        """

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = model(inputs)

        # Evaluate baseline
        # For first wp_n_epochs we take the combination of baseline and ema for previous batches
        # after that we take a slice of precomputed baseline values
        bl_val = bl_vals[num_batch] if bl_vals is not None else baseline.eval(inputs, cost)

        # Calculate loss
        reinforce_loss = torch.mean((cost - bl_val.detach()) * log_likelihood)

        return reinforce_loss, torch.mean(cost)

    def grad(model, inputs, baseline, num_batch):
        """Calculate gradients
        """
        loss, cost = rein_loss(model, inputs, baseline, num_batch)
        loss.backward()
        return loss, cost
        
    # For plotting
    train_loss_results = []
    train_cost_results = []
    val_cost_avg = []

    # Training loop
    for epoch in range(start_epoch, end_epoch):

        # Create dataset on current epoch
        data = generate_data_onfly(num_samples=samples, graph_size=graph_size)

        epoch_loss_avg_aux = []
        epoch_cost_avg_aux = []

        # Skip warm-up stage when we continue training from checkpoint
        if from_checkpoint and baseline.alpha != 1.0:
            print('Skipping warm-up mode')
            baseline.alpha = 1.0

        # If epoch > wp_n_epochs then precompute baseline values for the whole dataset else None
        bl_vals = baseline.eval_all(data)  # (samples, ) or None
        bl_vals = torch.reshape(bl_vals, (-1, batch)) if bl_vals is not None else None # (n_batches, batch) or None

        print("Current decode type: {}".format(model_torch.decode_type))
        
        train_batches = FastTensorDataLoader(data[0],data[1],data[2], batch_size=batch, shuffle=False)
        
        for num_batch, x_batch in tqdm(enumerate(train_batches), desc="batch calculation at epoch {}".format(epoch)):

            # Optimize the model
            loss_value, cost_val = grad(model_torch, x_batch, baseline, num_batch)

            # Clip gradients by grad_norm_clipping
            init_global_norm = torch.linalg.norm(model_torch.parameters)
            torch.nn.utils.clip_grad_norm_(model_torch.parameters(), grad_norm_clipping)
            global_norm = torch.linalg.norm(model_torch.parameters)

            if num_batch%batch_verbose == 0:
                print("grad_global_norm = {}, clipped_norm = {}".format(init_global_norm.numpy(), global_norm.numpy()))

            optimizer.step()

            # Track progress
            epoch_loss_avg_aux.append(loss_value)
            epoch_cost_avg_aux.append(cost_val)
            
            epoch_loss_avg = torch.mean(torch.tensor(epoch_loss_avg_aux))
            epoch_cost_avg = torch.mean(torch.tensor(epoch_cost_avg_aux))

            if num_batch%batch_verbose == 0:
                print("Epoch {} (batch = {}): Loss: {}: Cost: {}".format(epoch, num_batch, epoch_loss_avg, epoch_cost_avg))

        # Update baseline if the candidate model is good enough. In this case also create new baseline dataset
        baseline.epoch_callback(model_torch, epoch)
        set_decode_type(model_torch, "sampling")

        # Save model weights
        torch.save(model_torch.state_dict(),'model_checkpoint_epoch_{}_{}'.format(epoch, filename))

        # Validate current model
        val_cost = validate(validation_dataset, model_torch, val_batch_size)
        val_cost_avg.append(val_cost)

        train_loss_results.append(epoch_loss_avg)
        train_cost_results.append(epoch_cost_avg)

        pd.DataFrame(data={'epochs': list(range(start_epoch, epoch+1)),
                           'train_loss': [x.numpy() for x in train_loss_results],
                           'train_cost': [x.numpy() for x in train_cost_results],
                           'val_cost': [x.numpy() for x in val_cost_avg]
                           }).to_csv('backup_results_' + filename + '.csv', index=False)

        print(get_cur_time(), "Epoch {}: Loss: {}: Cost: {}".format(epoch, epoch_loss_avg, epoch_cost_avg))

    # Make plots and save results
    filename_for_results = filename + '_start={}, end={}'.format(start_epoch, end_epoch)
    get_results([x.numpy() for x in train_loss_results],
                [x.numpy() for x in train_cost_results],
                [x.numpy() for x in val_cost_avg],
                save_results=True,
                filename=filename_for_results,
                plots=True)
    