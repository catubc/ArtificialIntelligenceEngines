# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:25:46 2020

@author: gn-00
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


###############################################
class hopfield:
    def __init__(self, input_shape):
        self.train_data = []
        self.W = np.zeros([input_shape[0]*input_shape[1],input_shape[0]*input_shape[1]],dtype=np.int8)
        
    def addTrain(self,img_dir):
        
        # load 3 color data
        img = plt.imread(img_dir)
        img = np.mean(img,axis=2)
        if img.shape != (32,32):
            print('Error: image shape ',img.shape,' is not (32,32)')
            return
        img_mean = np.mean(img)
        img = np.where(img < img_mean,-1,1)
        train_data = img.flatten()
        
        #
        if True:
            self.W = self.W + np.outer(train_data,train_data) # change the weights to reflect the correlation between pixels
            
            #
            self.W[np.diag_indices(train_data.shape[0])] = 0
        #
        
        # old slow method
        else:
            for i in trange(train_data.size, desc='making weight matrix'):
                for j in range(i,train_data.size):
                    if i==j:
                        self.W[i][j] = 0
                    else:
                        w_ij = train_data[i]*train_data[j]
                        self.W[i][j] += w_ij
                        self.W[j][i] += w_ij
                
    #                
    def update(self,state,idx=None):
        if idx==None:
            # state = np.matmul(self.W,state)
            # state = np.where(state<0,-1,1)
            new_state = np.matmul(self.W,state)
            #new_state[new_state < 0] = -1
            #new_state[new_state > 0] = 1
            #new_state[new_state == 0] = state[new_state == 0]
            state = new_state
        else:
            # state[idx] = np.matmul(self.W[idx],state)
            # state[idx] = np.where(state[idx] < 0,-1,1)
            new_state = np.matmul(self.W[idx],state)
            if new_state < 0:
                state[idx] = -1
            elif new_state > 0:
                state[idx] = 1
        return state


    #
    def predict_no_plot(self,
						mat_input,
						iteration,
	                    asyn=False,
                        async_iteration=200):
        #
        input_shape = mat_input.shape
        mat_input = np.where(mat_input < 0.5,-1,1)
        e_list = []

        e = self.energy(mat_input.flatten())
        e_list.append(e)
        state = mat_input.flatten()
        
        #
        states = [] # keep track of all updates
        if asyn:
            print('Starting asynchronous update with ',iteration,' iterations')
            for i in range(iteration):
                idxes = np.random.choice(np.arange(state.size), 
                                          state.size, replace=False)
                #
                for idx in tqdm(idxes):   
                    state = self.update(state,idx)
                    state_show = np.where(state < 1,0,1).reshape(input_shape)
                    states.append(state.copy())

                new_e = self.energy(state) #-0.5*np.matmul(np.matmul(state.T,self.W),state)
                print('Iteration#',i,', Energy: ',new_e)
                if new_e == e:
                    print('Energy remain unchanged, update will now stop.')
                    break
                e = new_e
                e_list.append(e)
        #
        else:
            print('Starting synchronous update with ',iteration,' iterations')
            states = []
            states.append(state.copy())
            for i in range(iteration):
                state = self.update(state)
                state = np.where(state < 1, 0,1)
                states.append(state.copy())

                new_e = self.energy(state)

                print('Iteration#',i,', Energy: ',new_e)
                if new_e == e:
                    print('Energy remain unchanged, update will now stop.')
                    break
                e = new_e
                e_list.append(e)
        print('Iteration completed, update will now stop.')
        

        return np.where(state < 1,0,1).reshape(input_shape),e_list, states
        
        
    #
    def predict(self,mat_input,
                     original_img,
                     iteration,
                     asyn=False,
                     async_iteration=200):
        input_shape = mat_input.shape
        fig,axs = plt.subplots(1,3)
        #axs[0].axis('off')
        axs[1].imshow(original_img, cmap='binary')
        axs[1].set_title("original img")
        
        print(input_shape)
        graph = axs[0].imshow(mat_input*255,cmap='binary')
        mat_input = np.where(mat_input < 0.5,-1,1)
        fig.canvas.draw_idle()
        plt.pause(1)
        e_list = []
        
        e = self.energy(mat_input.flatten())
        e_list.append(e)
        state = mat_input.flatten()
        
        #
        if asyn:
            states =[]
            print('Starting asynchronous update with ',iteration,' iterations')
            for i in range(iteration):
                idxes = np.random.choice(np.arange(state.size), 
                                          state.size, replace=False)
                #for j in range(async_iteration):
                for idx in tqdm(idxes):   
                    #idx = np.random.randint(state.size)
                    state = self.update(state,idx)
                    state_show = np.where(state < 1,0,1).reshape(input_shape)
                    graph.set_data(state_show*255)
                    axs[0].set_title('Async update Iteration #%i' %i)
                    fig.canvas.draw_idle()
                    #plt.pause(0.01)
                new_e = -0.5*np.matmul(np.matmul(state.T,self.W),state)
                print('Iteration#',i,', Energy: ',new_e)
                if new_e == e:
                    print('Energy remain unchanged, update will now stop.')
                    break
                e = new_e
                e_list.append(e)
        #
        else:
            print('Starting synchronous update with ',iteration,' iterations')
            states = []
            states.append(state.copy())
            for i in range(iteration):
                state = self.update(state)
                state = np.where(state < 1, 0,1)
                states.append(state.copy())
                state_show = state.copy().reshape(input_shape)
                graph.set_data(state_show*255)
                axs[0].set_title('Sync update Iteration #%i' %i)
                fig.canvas.draw_idle()
                plt.pause(0.5)
                new_e = self.energy(state)
                #new_e = -0.5*np.matmul(np.matmul(state.T,self.W),state)
                print('Iteration#',i,', Energy: ',new_e)
                if new_e == e:
                    print('Energy remain unchanged, update will now stop.')
                    break
                e = new_e
                e_list.append(e)
        print('Iteration completed, update will now stop.')
        
        # show residual
        im2 = axs[2].imshow(state_show-original_img, cmap='binary')
        axs[2].set_title("residual difference")
        #im = ax.imshow(data, cmap='binary')
        fig.colorbar(im2)
        
        plt.pause(2)
        plt.close()
        return np.where(state < 1,0,1).reshape(input_shape),e_list, states
    
    def energy(self,o):
        e = -0.5*np.matmul(np.matmul(o.T,self.W),o)
        return e
