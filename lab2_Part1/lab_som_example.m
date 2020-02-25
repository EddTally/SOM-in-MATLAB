function som = lab_som_example (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius, mute)
% som = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% -- Purpose: Trains a 1D SOM i.e. A SOM where the neurons are arranged
%             in a single line. 
%             
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <neuronCount> number of neurons 
% -- <trainingSteps> number of training steps 
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius
% -- <mute> do not print on the screen

% TODO:
% The student will need to complete this function so that it returns
% a matrix 'som' containing the weights of the trained SOM.
% The weight matrix should be arranged as follows, where
% N is the number of features and M is the number of neurons:
%
% Neuron1_Weight1 Neuron1_Weight2 ... Neuron1_WeightN
% Neuron2_Weight1 Neuron2_Weight2 ... Neuron2_WeightN
% ...
% NeuronM_Weight1 NeuronM_Weight2 ... NeuronM_WeightN
%
% It is important that this format is maintained as it is what
% lab_vis(...) expects.
%
% Some points that you need to consider are:
%   - How should you randomise the weight matrix at the start?
%   - How do you decay both the learning rate and radius over time?
%   - How does updating the weights of a neuron effect those nearby?
%   - How do you calculate the distance of two neurons when they are
%     arranged on a single line?
% 

    % startLearningRate should be equal to 0.1 (heuristic)
    learning0 = startLearningRate;
    % startRadius should be equal to max(latticeHeight,latticeWidth)/2
    sigma0 = startRadius;
    
    % I do not found infos about T1.
    T1 = 1000;%trainingSteps/log(learning0);
    % T2 is dependent by the number of iterations. 
    T2 = trainingSteps/log(sigma0);

    % 1. Initialize weights to some small, random values.
    [datas features] = size(trainingData);
    som = rand(neuronCount,features);
    
    currentLearning = learning0;
    currentSigma = sigma0;
    
    % 2. Repeat until convergence
    for t=1:trainingSteps
        
        % 2a. Select the next input pattern from the database
        xn = trainingData(randi(datas,1,1),:);
        %disp(['Input data choosen: ' mat2str(xn)]);
            
        % 2a1. Find the unit wi that best matches the input pattern
        BMU = findWinnerNeuron(xn,som);
        
        for n=1:neuronCount
        
            % 2a2. Update the weights of the winner BMU and all its neighbors wk
            distanceInfluence(t*n) = exp(-(abs(oneDimensionalDistance(BMU,n))/(2*(currentSigma(t)))));
            som(n,:) = som(n,:) + (currentLearning(t)*distanceInfluence(t*n).*(xn-som(n,:)));
            
        end
        
        % 2b. Update the learning rate decay rule
        currentLearning(t+1) = learning0 * exp(-t/T1);
        
        % 2c. Update the neighborhood size decay rule
        currentSigma(t+1) = sigma0 * exp(-t/T2);
        
        
        if(mod(t,500)==0 && mute~=1)
            lab_vis(som,trainingData);
            drawnow;
            disp(['Iteration ' num2str(t) ' : completed']);
        end
        
    end
    
    if(mute~=1)
        %subplot(4,1,1);
        lab_vis(som,trainingData);
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
    for n=2:numberOfNeurons
        matchingScore = norm(xn-SOM(n,:),2);
        if matchingScore<minMatchingScore
            minMatchingScore = matchingScore;
            winner = n;
        end
    end
end

% Compute 1-dimensional distance
function distance1 = oneDimensionalDistance (vect1, vect2)
    distance1 = vect1 - vect2;
end