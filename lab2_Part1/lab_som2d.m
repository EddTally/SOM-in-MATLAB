function [som,grid] = lab_som2d (trainingData, neuronCountW, neuronCountH, trainingSteps, startLearningRate, startRadius)
% som = lab_som2d (trainingData, neuronCountW, neuronCountH, trainingSteps, startLearningRate, startRadius)
% -- Purpose: Trains a 2D SOM, which consists of a grid of
%             (neuronCountH * neuronCountW) neurons.
%             
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <grid> returns the location of the neurons in the grid
% -- <neuronCountW> number of neurons along width
% -- <neuronCountH> number of neurons along height
% -- <trainingSteps> number of training steps 
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius used to specify the initial neighbourhood size
%

% TODO:
% The student will need to copy their code from lab_som() and
% update it so that it uses a 2D grid of neurons, rather than a 
% 1D line of neurons.
% 
% Your function will still return the a weight matrix 'som' with
% the same format as described in lab_som().
%
% However, it will additionally return a vector 'grid' that will
% state where each neuron is located in the 2D SOM grid. 
% 
% grid(n, :) contains the grid location of neuron 'n'
%
% For example, if grid = [[1,1];[1,2];[2,1];[2,2]] then:
% 
%   - som(1,:) are the weights for the neuron at position x=1,y=1 in the grid
%   - som(2,:) are the weights for the neuron at position x=2,y=1 in the grid
%   - som(3,:) are the weights for the neuron at position x=1,y=2 in the grid 
%   - som(4,:) are the weights for the neuron at position x=2,y=2 in the grid
%
% It is important to return the grid in the correct format so that
% lab_vis2d() can render the SOM correctly
Td = trainingData;
NcW = neuronCountW;
NcH = neuronCountH;
Ts = trainingSteps;
Slr = startLearningRate;
Srad = startRadius;

% Initilizing stuff
[dataLength, features] = size(trainingData);
totalNeurons = NcW*NcH;
som = randi([1 50], totalNeurons, features);

currentLearning = Slr;
currentSigma = Srad;
t1 = Ts/log(currentLearning);
t2 = Ts/log(currentSigma);



%%%%%%%%%%%%% Learning algorithm

    for i=1:neuronCountH
        for j=1:neuronCountW
            pos = ((i-1)*neuronCountW)+(j);
            grid(pos,:) = [i j];
        end
    end

    for t=1:Ts

        % 2a. Select the next input pattern from the database
        xn = Td(randi(dataLength,1,1),:); 
        
        winner = getMinNeuron(xn, som, totalNeurons);
        
        %Updating neuron weights
        for n = 1:totalNeurons
            som(n,:) = som(n,:)...
                + currentLearning(t)*nKern(grid(winner,:),grid(n,:), currentSigma(t))...
                .*(xn - som(n,:));
        end
    
        % 2b. Update the learning rate decay rule
        currentLearning(t+1) = Slr * exp(-t/t1);
    
        % 2c. Update the neighborhood size decay rule
        currentSigma(t+1) = Srad * exp(-t/t2);
    end

        lab_vis2d(som,grid,trainingData);
    end

% neighbourhood kernel function
function nKern = nKern(x, y, sigma)
    d = abs(eucdist(x,y)^2);
    nKern = exp(-(d)/(2*sigma^2));
end

%Finding minimum neuron
function winner = getMinNeuron(xn, som, Nc)
    minScore = norm(xn - som(1, :));
    winner = 1;
    for j = 2:Nc
        score = norm(xn - som(j, :));
        if minScore>score
            minScore = score;
            winner = j;
        end
    end
end


