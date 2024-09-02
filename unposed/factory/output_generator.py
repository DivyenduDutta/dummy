import logging
import time
import pandas as pd
import torch

from utils.save_load import save_test_results
from utils.others import dict_to_device

logger = logging.getLogger(__name__)

class Output_Generator:
    def __init__(self, model, dataloader, save_dir, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.device = device

        self.result = pd.DataFrame()
        self.pred_pose = torch.Tensor().to(device)

    def generate(self):
        logger.info("Generating outputs started.")
        self.model.eval()
        #time0 = time.time()
        self.__generate()
        #logger.info('Generating outputs is completed in: %.2f' % (time.time() - time0))
        save_test_results(self.result, [self.pred_pose], self.save_dir)

    # original version of the function
    '''
    def __generate(self):
        for data in self.dataloader:
            with torch.no_grad():
                # predict & calculate loss
                model_outputs = self.model(dict_to_device(data, self.device))
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
                pred_pose = model_outputs['pred_pose']
    
                self.store_results(pred_pose)
    '''
                    
    # this version of the function is used for benchmarking but time for each batch is printed separately
    '''
    def __generate(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        num_total_runs = 5
        num_total_frames_pred = 0
        batch_execution_times = dict() #stores the time to predict all frames for all batches and for all runs
        for run_num in range(num_total_runs):
            batch_num = 0
            for data in self.dataloader:
                #print('\n')
                #print(data.keys())
                #print(data['observed_pose'].shape)
                
                # checking if removing ['observed_metric_pose', 'future_metric_pose', 'future_pose', 'video_section']
                # from the input data matters or not -- START
                del data['observed_metric_pose']
                del data['future_metric_pose']
                del data['future_pose']
                del data['video_section']
                # END - doesnt matter, all of the above are not needed for the prediction

                if batch_num not in batch_execution_times.keys():
                    batch_execution_times[batch_num] = list()

                time0 = time.time()
                with torch.no_grad():
                    # predict & calculate loss
                    #time0 = time.time()
                    model_outputs = self.model(dict_to_device(data, self.device))
                    assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
                    pred_pose = model_outputs['pred_pose']
                    #logger.info('Generating output is completed in: %.2f' % (time.time() - time0))

                    self.store_results(pred_pose)
                    logger.info('Generating output is completed in: %.5f' % (time.time() - time0))
                    num_total_frames_pred = pred_pose.shape[1]
                    #logger.info('total number of frames being predicted: %.5f' % (pred_pose.shape[1]))
                    all_frames_time = (time.time() - time0)
                    # doesnt make sense to consider per frame since as per this model outputs predictions for all frames in one go ie, non autoregressive
                    #batch_execution_times[batch_num].append(all_frames_time/pred_pose.shape[1])
                    batch_execution_times[batch_num].append(all_frames_time*1000) #convert to ms
                    
                batch_num += 1
                    
                # my change
                #break

        # plot the execution times in a graph
        x_values = range(num_total_runs)
        pyplot_lines = []
        batch_avg_execution_times = []
        for batch_num_key in batch_execution_times.keys():
            # comma used below with intention, see - https://stackoverflow.com/questions/11983024/matplotlib-legends-not-working
            line, = plt.plot(x_values, batch_execution_times[batch_num_key], marker='o', linestyle='-')
            pyplot_lines.append(line)
            batch_avg_execution_times.append(f'{sum(batch_execution_times[batch_num_key])/len(batch_execution_times[batch_num_key]):.5f} ms')
        plt.xlabel('runs')
        plt.ylabel(f'time to predict {num_total_frames_pred} frames in ms')
        plt.title(f'time to predict {num_total_frames_pred} frames for {batch_num} batches (batch size: {1}) across {num_total_runs} runs')
        plt.legend(pyplot_lines, batch_avg_execution_times)
        plt.show()
        #for batch_num_key in batch_execution_times.keys():
        #    print(batch_execution_times[batch_num_key])
        
        #print('\nhere')
        #print(self.result.shape)
    '''
    
    # this version of the function is used for benchmarking avg time for all batches is printed. This I feel is better
    def __generate(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device: %s' % device)

        num_total_runs = 5
        num_total_frames_pred = 0
        batch_execution_times = list() #stores the avg time to predict all frames for all batches and for all runs
        
        for run_num in range(num_total_runs):
            batch_num_count = 0
            batch_execution_time = 0 #stores the time to predict all frames for 1 batch where each batch has 1 motion sequence ie, batch_size = 1
            for data in self.dataloader:
                if batch_num_count == 1: #computing time for just 1 batche
                    break
                #print('\n')
                #print(data.keys())
                #print(data['observed_pose'].shape)
                
                # checking if removing ['observed_metric_pose', 'future_metric_pose', 'future_pose', 'video_section']
                # from the input data matters or not -- START
                del data['observed_metric_pose']
                del data['future_metric_pose']
                del data['future_pose']
                del data['video_section']
                # END - doesnt matter, all of the above are not needed for the prediction

                print(f'Run num: {run_num+1} start')
                time0 = time.time()
                with torch.no_grad():
                    # predict & calculate loss
                    #time0 = time.time()
                    print('\nInput shape:')
                    print(data['observed_pose'].shape)
                    model_outputs = self.model(dict_to_device(data, self.device))
                    assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
                    pred_pose = model_outputs['pred_pose']
                    #logger.info('Generating output is completed in: %.2f' % (time.time() - time0))

                    self.store_results(pred_pose)
                    logger.info('Generating output is completed in: %.5f' % (time.time() - time0))
                    num_total_frames_pred = pred_pose.shape[1]
                    #logger.info('total number of frames being predicted: %.5f' % (pred_pose.shape[1]))
                all_frames_time = (time.time() - time0)
                print(f'Run num: {run_num+1} end')
                # doesnt make sense to consider per frame since as per this model outputs predictions for all frames in one go ie, non autoregressive
                #batch_execution_times[batch_num].append(all_frames_time/pred_pose.shape[1])
                batch_execution_time += all_frames_time
                batch_num_count += 1

            batch_execution_time /= batch_num_count #calculate avg over all batches
            batch_execution_times.append(batch_execution_time*1000) #convert to ms    
                # my change
                #break

        # plot the execution times in a graph
        x_values = range(1, num_total_runs+1)
        plt.figure(figsize=(10,6))
        print('\n Batch execution times')
        print(batch_execution_times)
        line, = plt.plot(x_values, batch_execution_times, marker='o', linestyle='-')
        plt.xlabel('Runs')
        plt.ylabel(f'Time to predict {num_total_frames_pred} frames in seconds')
        plt.title(f'Time to predict {num_total_frames_pred} frames across {num_total_runs} runs. Num of batches: {batch_num_count}. Batch size: 1')
        avg_time_excl_first = f'Avg time excl. first run : {(sum(batch_execution_times[1:])/len(batch_execution_times[1:])):.5f} ms'
        avg_time_incl_first = f'Avg time incl. first run : {(sum(batch_execution_times)/len(batch_execution_times)):.5f} ms'
        time_first_run = f'Time for first run : {(batch_execution_times[0]):.5f} ms'
        plt.legend([line, line, line], [avg_time_excl_first, avg_time_incl_first, time_first_run])
        plt.show()
        #for batch_num_key in batch_execution_times.keys():
        #    print(batch_execution_times[batch_num_key])
        
        #print('\nhere')
        #print(self.result.shape)
    
    
    # this version of the function measures the time for each batch rather than all put together                
    '''
    def __generate(self):
        for data in self.dataloader:
            #print('\n')
            #print(data.keys())
            #print(data['observed_pose'].shape)
            
            # checking if removing ['observed_metric_pose', 'future_metric_pose', 'future_pose', 'video_section']
            # from the input data matters or not -- START
            del data['observed_metric_pose']
            del data['future_metric_pose']
            del data['future_pose']
            del data['video_section']
            # END - doesnt matter, all of the above are not needed for the prediction

            time0 = time.time()
            with torch.no_grad():
                # predict & calculate loss
                #input_tensor = dict_to_device(data, self.device)
                #time0 = time.time()
                model_outputs = self.model(dict_to_device(data, self.device))
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
                pred_pose = model_outputs['pred_pose']
                #logger.info('Generating output is completed in: %.2f' % (time.time() - time0))

                self.store_results(pred_pose)
                logger.info('Generating output is completed in: %.5f' % (time.time() - time0))
                #logger.info('self.device: %s' % (self.device))
                
                
            # my change
            #break
        
        #print('\nhere')
        #print(self.result.shape)
    '''

    def store_results(self, pred_pose):
        print('\nPrediction shape:')
        print(pred_pose.shape)
        # update tensors
        self.pred_pose = torch.cat((self.pred_pose, pred_pose), 0)

        # to cpu
        if self.device == 'cuda':
            pred_pose = pred_pose.detach().cpu()
        # update dataframe
        for i in range(pred_pose.shape[0]):
            single_data = {'pred_pose': str(pred_pose[i].numpy().tolist())}
            self.result = self.result.append(single_data, ignore_index=True)
