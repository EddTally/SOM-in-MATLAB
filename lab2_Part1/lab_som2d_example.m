function [som,grid] = lab_som2d_example (trainingData, neuronCountW, neuronCountH, trainingSteps, startLearningRate, startRadius, mute)
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
% -- <startRadius> initial radius
% -- <mute> do not print on the screen

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

    
    % startLearningRate should be equal to 0.1 (heuristic)
    learning0 = startLearningRate;
    % startRadius should be equal to max(latticeHeight,latticeWidth)/2
    sigma0 = startRadius;
    
    % I do not found infos about T1.
    T1 = 1000;%trainingSteps/log(learning0);
    % T2 is dependent by the number of iterations. 
    T2 = 1000/log(sigma0);

    % 1. Initialize weights to some small, random values.
    [datas features] = size(trainingData);
    totalNeurons = neuronCountW*neuronCountH;
    som = rand(totalNeurons,features);
    
    for i=1:neuronCountH
        for j=1:neuronCountW
            pos = ((i-1)*neuronCountW)+(j);
            grid(pos,:) = [i j];
        end
    end
    
    currentLearning = learning0;
    currentSigma = sigma0;
    
    % 2. Repeat until convergence
    for t=1:trainingSteps
        
        % 2a. Select the next input pattern from the database
        xn = trainingData(randi(datas,1,1),:);
        %disp(['Input data choosen: ' mat2str(xn)]);
            
        % 2a1. Find the unit wi that best matches the input pattern
        BMU = findWinnerNeuron(xn,som);
        
        for n=1:totalNeurons
        
            % 2a2. Update the weights of the winner BMU and all its neighbors wk
            distanceInfluence(t*n) = exp(-((latticeDistance(grid(BMU,:),grid(n,:))^2)/(2*(currentSigma(t)^2)) ));
            som(n,:) = som(n,:) + (currentLearning(t)*distanceInfluence(t*n).*(xn-som(n,:)));
            
        end
        
        % 2b. Update the learning rate decay rule
        currentLearning(t+1) = learning0 * exp(-t/T1);
        
        % 2c. Update the neighborhood size decay rule
        currentSigma(t+1) = sigma0 * exp(-t/T2);
        
        
        %if(mod(t,1)==0 && mute~=1)
            lab_vis2d(som,grid,trainingData);
            drawnow;
            %disp(['Iteration ' num2str(t) ' : completed']);
        %end
        
    end
    
    if(mute~=1)
        %subplot(4,1,1);
        lab_vis2d(som,grid,trainingData,t);
        %subplot(4,1,2); title('Learning rate'); plot(currentLearning);
        %subplot(4,1,3); title('Neighborhood rate'); plot(currentSigma);
        %subplot(4,1,4); title('Distance Influence'); plot(distanceInfluence);
    end
end

% Find the position of the winner neuron
function winner = findWinnerNeuron (xn,SOM)
    [numberOfNeurons N] = size(SOM);
    minMatchingScore = SOM(1,:);
    winner = 1;
    for n=1:numberOfNeurons
        matchingScore = norm(xn-SOM(n,:),2);
        if matchingScore<minMatchingScore
            minMatchingScore = matchingScore;
            winner = n;
        end
    end
end

% Compute lattice distance
function distance1 = latticeDistance (vect1, vect2)
    distance1 = norm(vect1 - vect2,1);
end