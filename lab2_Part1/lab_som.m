function som = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% som = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% -- Purpose: Trains a 1D SOM i.e. A SOM where the neurons are arranged
%             in a single line.
%
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <neuronCount> number of neurons
% -- <trainingSteps> number of training steps
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius used to specify the initial neighbourhood size

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

Td = trainingData;
Nc = neuronCount;
Ts = trainingSteps;
Slr = startLearningRate;
Srad = startRadius;

% Initilizing stuff
[dataLength, features] = size(trainingData);
som = randi([1 50], Nc, features);

currentLearning = Slr;
currentSigma = Srad;
t1 = Ts/log(currentLearning);
t2 = Ts/log(currentSigma);



%%%%%%%%%%%%% Learning algorithm

    for t=1:Ts

        % 2a. Select the next input pattern from the database
        xn = Td(randi(dataLength,1,1),:); 
        
        winner = getMinNeuron(xn, som, Nc);
        
        %Updating neuron weights
        for n = 1:Nc
            som(n,:) = som(n,:)...
                + currentLearning(t)*nKern(winner,n, currentSigma(t))...
                .*(xn - som(n,:));
        end
    
        % 2b. Update the learning rate decay rule
        currentLearning(t+1) = Slr * exp(-t/t1);
    
        % 2c. Update the neighborhood size decay rule
        currentSigma(t+1) = Srad * exp(-t/t2);
    end

        lab_vis(som,trainingData);
    end

% neighbourhood kernel function
function nKern = nKern(x, y, sigma)
    d = abs(eucdist(x,y));
    nKern = exp(-(d)/(2*sigma));
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

