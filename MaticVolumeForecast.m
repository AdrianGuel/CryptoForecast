
clearvars
close all
file=sprintf('datamatic.csv'); %we upload the data
[timestamp,volume] = csvimport(file, 'columns', {'timestamp','volume'});
% 
Ts=1; %sampling period in minutes
% data=zeros(1,length(dataaux)/Ts);
% %% Generate vector of information sampled every Ts minutes
% j=1;
% for i=1:length(dataaux)
%     if mod(i,Ts)==0
%         data(j)=dataaux(i);
%         j=j+1;
%     end
% end

volume=volume';
%% Training
numTimeStepsTrain = floor(0.45*numel(volume));%Train with 10% of the data
dataTrain = volume(1:numTimeStepsTrain+1);
dataTest = volume(numTimeStepsTrain+1:end);

mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain- mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 500;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);
dataTestStandardized = (dataTest- mu) / sig;
XTest = dataTestStandardized(1:end-1);
%net = predictAndUpdateState(net,XTrain);
YTest = dataTest(2:end);

net = resetState(net);
net = predictAndUpdateState(net,XTrain);
%% PDF initialization
pdf=zeros(numel(XTest)+1,100);
xi=zeros(numel(XTest)+1,100);
[pdf(1,:),xi(1,:)]  = ksdensity(XTrain(end-24:end));
Energy=zeros(length(pdf(:,1)),1);

%% Forecasting PDFs
figure
set(gcf,'color','w');
hold on
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(i)] = predictAndUpdateState(net,XTest(i),'ExecutionEnvironment','cpu');
    XTrain(1)=[];
    XTrain(end+1)=YPred(i);
    [pdf(i+1,:),xi(i+1,:)]  = ksdensity(XTrain(end-24:end));
    plot3(i*ones(1,100)./24,xi(i,:),pdf(i,:),'color',[0.4 0.6 0.7])
    for j=1:length(pdf(i,:))
        Energy(i)=Energy(i)+4*sqrt(pdf(i,j))*((sqrt(pdf(i+1,j))-sqrt(pdf(i,j)))^2)/(Ts^2); %equation (3.7)
    end
end
xlabel('Days','Interpreter','Latex','FontSize', 16)
ylabel('x','Interpreter','Latex','FontSize', 16)
zlabel('p(x,t)','Interpreter','Latex','FontSize', 16)
grid on
v = [-5 -2 5];
[caz,cel] = view(v);
%% Information length computation
figure
set(gcf,'color','w');
IL=zeros(length(Energy),1);
for i=2:length(Energy)
    IL(i)=IL(i-1)+sqrt(Energy(i)); %equation (3.8)
end
subplot(2,1,1)
tE=1:1:length(Energy);
    plot(tE./(24),Energy,'r','LineWidth',2);
        xlabel('Days','Interpreter','Latex','FontSize', 10)
        ylabel('$\mathcal{E}$ (Information velocity)','Interpreter','Latex','FontSize', 10)
    grid on
subplot(2,1,2)
tL=1:1:length(IL);
    plot(tL./(24),IL,'r','LineWidth',2);
    xlabel('Days','Interpreter','Latex','FontSize', 10)
    ylabel('$\mathcal{L}$ (Information length)','Interpreter','Latex','FontSize', 10)
    grid on

%% Plotting Results Forecasting
YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2));
tY=1:1:length(YTest);
figure
set(gcf,'color','w');
subplot(2,1,1)
plot(tY./(24),YTest)
hold on
plot(tY./(24),YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel('gigawatts','Interpreter','Latex','FontSize', 16)
title("Forecast with Updates")

subplot(2,1,2)
stem(tY./(24),YPred - YTest)
ylabel('gigawatts','Interpreter','Latex','FontSize', 16)
ylabel("Error")
title("RMSE = " + rmse)