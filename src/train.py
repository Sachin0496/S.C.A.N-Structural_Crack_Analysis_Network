# --- Import necessary libraries ---

# Local imports for data handling (augmentation, dataset loading, and pipelines)
from data.augmentation import augCompose, RandomBlur, RandomColorJitter
from data.dataset import readIndex, dataReadPip, loadedDataset

# TQDM for creating progress bars
from tqdm import tqdm

# Local import for the DeepCrack model architecture
from model.deepcrack import DeepCrack

# Local import for the custom trainer class
from trainer import DeepCrackTrainer

# Local import for configuration settings
from config import Config as cfg

# Standard libraries
import numpy as np
import torch
import os
import cv2
import sys

# --- GPU Configuration ---
# Set which GPU(s) to make visible to the script based on the config file
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


def main():
    """
    Main function to run the training and validation process.
    """
    
    # ----------------------- 1. Dataset Setup ----------------------- #
    
    # Define the data augmentation operations
    # Here, we randomly apply color jitter (with 50% probability) and random blur (with 20% probability)
    data_augment_op = augCompose(transforms=[[RandomColorJitter, 0.5], [RandomBlur, 0.2]])

    # Create the data processing pipeline for the training set, including augmentations
    train_pipline = dataReadPip(transforms=data_augment_op)
    
    # Create the data processing pipeline for the test/validation set (no augmentations)
    test_pipline = dataReadPip(transforms=None)

    # Load the training dataset
    # readIndex reads the file paths from the training data path
    # loadedDataset creates a PyTorch Dataset object using the training pipeline
    train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=True), preprocess=train_pipline)

    # Load the test/validation dataset
    test_dataset = loadedDataset(readIndex(cfg.test_data_path), preprocess=test_pipline)

    # Create the PyTorch DataLoader for the training set
    # This handles batching, shuffling, and multi-threaded data loading
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)
    
    # Create the PyTorch DataLoader for the validation set
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.val_batch_size,
                                             shuffle=False, num_workers=4, drop_last=True)

    # -------------------- 2. Build Model & Trainer --------------------- #

    # Set the device to CUDA (GPU)
    device = torch.device("cuda")
    # Get the number of available GPUs
    num_gpu = torch.cuda.device_count()

    # Instantiate the DeepCrack model
    model = DeepCrack()
    
    # Wrap the model with DataParallel to use multiple GPUs
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    # Move the model to the GPU(s)
    model.to(device)

    # Instantiate the custom trainer class, passing in the model
    # The trainer likely handles the optimizer, loss functions, and logging
    trainer = DeepCrackTrainer(model).to(device)

    # -------------------- 3. Load Pretrained Model (if specified) --------------------- #
    
    if cfg.pretrained_model:
        # Load the saved weights (state dictionary) from the checkpoint file
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        # Get the current model's state dictionary
        model_dict = model.state_dict()

        # Filter the loaded dictionary to only include keys that are present in the current model
        # This prevents errors if the checkpoint has a different architecture
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # Update the current model's dictionary with the loaded weights
        model_dict.update(pretrained_dict)
        # Load the updated state dictionary back into the model
        model.load_state_dict(model_dict)
        
        # Log that the checkpoint was successfully loaded
        trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, 'train info')

    # -------------------- 4. Main Training Loop --------------------- #
    
    try:
        # Loop over the specified number of epochs
        for epoch in range(1, cfg.epoch):
            trainer.vis.log('Start Epoch %d ...' % epoch, 'train info')
            # Set the model to training mode (e.g., enable dropout, batch norm updates)
            model.train()

            # --- Training ---
            # Create a TQDM progress bar for the training loader
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            
            # Iterate over batches of training data
            for idx, (img, lab) in bar:
                # Move image and label tensors to the GPU
                data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                
                # Perform a single training step (forward pass, loss, backward pass, optimizer step)
                # The 'train_op' method is defined in the DeepCrackTrainer class
                pred = trainer.train_op(data, target)
                
                # --- Log Training Loss (periodically) ---
                if idx % cfg.vis_train_loss_every == 0:
                    # Log scalar loss values
                    trainer.vis.log(trainer.log_loss, 'train_loss')
                    # Plot the losses (total loss and loss for each fusion layer)
                    trainer.vis.plot_many({
                        'train_total_loss': trainer.log_loss['total_loss'],
                        'train_output_loss': trainer.log_loss['output_loss'],
                        'train_fuse5_loss': trainer.log_loss['fuse5_loss'],
                        'train_fuse4_loss': trainer.log_loss['fuse4_loss'],
                        'train_fuse3_loss': trainer.log_loss['fuse3_loss'],
                        'train_fuse2_loss': trainer.log_loss['fuse2_loss'],
                        'train_fuse1_loss': trainer.log_loss['fuse1_loss'],
                    })

                # --- Log Training Accuracy (periodically) ---
                if idx % cfg.vis_train_acc_every == 0:
                    # Calculate accuracy metrics
                    trainer.acc_op(pred[0], target)
                    # Log accuracy values
                    trainer.vis.log(trainer.log_acc, 'train_acc')
                    # Plot accuracy metrics
                    trainer.vis.plot_many({
                        'train_mask_acc': trainer.log_acc['mask_acc'],
                        'train_mask_pos_acc': trainer.log_acc['mask_pos_acc'],
                        'train_mask_neg_acc': trainer.log_acc['mask_neg_acc'],
                    })
                
                # --- Log Training Images (periodically) ---
                if idx % cfg.vis_train_img_every == 0:
                    # Log images: input, final output, ground truth, and intermediate fusion layer outputs
                    trainer.vis.img_many({
                        'train_img': data.cpu(),
                        'train_output': torch.sigmoid(pred[0].contiguous().cpu()), # Apply sigmoid to get probabilities
                        'train_lab': target.unsqueeze(1).cpu(),
                        'train_fuse5': torch.sigmoid(pred[1].contiguous().cpu()),
                        'train_fuse4': torch.sigmoid(pred[2].contiguous().cpu()),
                        'train_fuse3': torch.sigmoid(pred[3].contiguous().cpu()),
                        'train_fuse2': torch.sigmoid(pred[4].contiguous().cpu()),
                        'train_fuse1': torch.sigmoid(pred[5].contiguous().cpu()),
                    })

                # -------------------- 5. Validation Loop (periodically) --------------------- #
                
                if idx % cfg.val_every == 0:
                    trainer.vis.log('Start Val %d ....' % idx, 'train info')
                    
                    # --- Validation ---
                    # Set the model to evaluation mode (e.g., disable dropout, use running averages for batch norm)
                    model.eval()
                    
                    # Initialize dictionaries to accumulate validation loss and accuracy
                    val_loss = {
                        'eval_total_loss': 0,
                        'eval_output_loss': 0,
                        'eval_fuse5_loss': 0,
                        'eval_fuse4_loss': 0,
                        'eval_fuse3_loss': 0,
                        'eval_fuse2_loss': 0,
                        'eval_fuse1_loss': 0,
                    }
                    val_acc = {
                        'mask_acc': 0,
                        'mask_pos_acc': 0,
                        'mask_neg_acc': 0,
                    }

                    bar.set_description('Epoch %d --- Evaluation --- :' % epoch)
                    
                    # Disable gradient calculations to save memory and speed up validation
                    with torch.no_grad():
                        # Iterate over the validation data
                        for idx, (img, lab) in enumerate(val_loader, start=1):
                            # Move validation data to the GPU
                            val_data, val_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(
                                torch.cuda.FloatTensor).to(device)
                            
                            # Perform a forward pass (no backward pass)
                            # The 'val_op' is defined in the DeepCrackTrainer class
                            val_pred = trainer.val_op(val_data, val_target)
                            # Calculate accuracy
                            trainer.acc_op(val_pred[0], val_target)
                            
                            # Accumulate losses and accuracies from the trainer's log
                            val_loss['eval_total_loss'] += trainer.log_loss['total_loss']
                            val_loss['eval_output_loss'] += trainer.log_loss['output_loss']
                            val_loss['eval_fuse5_loss'] += trainer.log_loss['fuse5_loss']
                            val_loss['eval_fuse4_loss'] += trainer.log_loss['fuse4_loss']
                            val_loss['eval_fuse3_loss'] += trainer.log_loss['fuse3_loss']
                            val_loss['eval_fuse2_loss'] += trainer.log_loss['fuse2_loss']
                            val_loss['eval_fuse1_loss'] += trainer.log_loss['fuse1_loss']
                            val_acc['mask_acc'] += trainer.log_acc['mask_acc']
                            val_acc['mask_pos_acc'] += trainer.log_acc['mask_pos_acc']
                            val_acc['mask_neg_acc'] += trainer.log_acc['mask_neg_acc']
                        
                        # --- This 'else' block executes *after* the validation 'for' loop finishes ---
                        else:
                            # Log a batch of validation images (from the last batch)
                            trainer.vis.img_many({
                                'eval_img': val_data.cpu(),
                                'eval_output': torch.sigmoid(val_pred[0].contiguous().cpu()),
                                'eval_lab': val_target.unsqueeze(1).cpu(),
                                'eval_fuse5': torch.sigmoid(val_pred[1].contiguous().cpu()),
                                'eval_fuse4': torch.sigmoid(val_pred[2].contiguous().cpu()),
                                'eval_fuse3': torch.sigmoid(val_pred[3].contiguous().cpu()),
                                'eval_fuse2': torch.sigmoid(val_pred[4].contiguous().cpu()),
                                'eval_fuse1': torch.sigmoid(val_pred[5].contiguous().cpu()),

                            })
                            
                            # Calculate and plot average validation losses
                            trainer.vis.plot_many({
                                'eval_total_loss': val_loss['eval_total_loss'] / idx,
                                'eval_output_loss': val_loss['eval_output_loss'] / idx,
                                'eval_fuse5_loss': val_loss['eval_fuse5_loss'] / idx,
                                'eval_fuse4_loss': val_loss['eval_fuse4_loss'] / idx,
                                'eval_fuse3_loss': val_loss['eval_fuse3_loss'] / idx,
                                'eval_fuse2_loss': val_loss['eval_fuse2_loss'] / idx,
                                'eval_fuse1_loss': val_loss['eval_fuse1_loss'] / idx,

                            })
                            
                            # Calculate and plot average validation accuracies
                            trainer.vis.plot_many({
                                'eval_mask_acc': val_acc['mask_acc'] / idx,
                                'eval_mask_neg_acc': val_acc['mask_neg_acc'] / idx,
                                'eval_mask_pos_acc': val_acc['mask_pos_acc'] / idx,

                            })
                            
                            # ----------------- 6. Save Best Model ---------------- #
                            # Check if the current model is the best one based on validation accuracy
                            if cfg.save_pos_acc < (val_acc['mask_pos_acc'] / idx) and cfg.save_acc < (
                                    val_acc['mask_acc'] / idx):
                                # Update the best accuracy trackers in the config
                                cfg.save_pos_acc = (val_acc['mask_pos_acc'] / idx)
                                cfg.save_acc = (val_acc['mask_acc'] / idx)
                                # Save the model checkpoint with epoch and accuracy in the filename
                                trainer.saver.save(model, tag='%s_epoch(%d)_acc(%0.5f/%0.5f)' % (
                                    cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc))
                                # Log that a new best model was saved
                                trainer.vis.log('Save Model %s_epoch(%d)_acc(%0.5f/%0.5f)' % (
                                    cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc), 'train info')

                    # Set the progress bar description back to 'Training'
                    bar.set_description('Epoch %d --- Training --- :' % epoch)
                    # Set the model *back* to training mode
                    model.train()

            # --- Save model at the end of every epoch ---
            if epoch != 0:
                trainer.saver.save(model, tag='%s_epoch(%d)' % (
                    cfg.name, epoch))
                trainer.vis.log('Save Model -%s_epoch(%d)' % (
                    cfg.name, epoch), 'train info')

    # -------------------- 7. Handle KeyboardInterrupt --------------------- #
    
    except KeyboardInterrupt:
        # If training is manually stopped (Ctrl+C)
        # Save the current model state automatically
        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        trainer.vis.log('Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name,
                        'train info')
        trainer.vis.log('Training End!!')
        
        # Exit the script
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


# --- Standard Python entry point ---
if __name__ == '__main__':
    main()
