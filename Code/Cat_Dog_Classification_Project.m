%% Advanced Cat vs Dog Image Classification Project
% Deep Learning with Data Augmentation & Fine-Tuning
% 2nd Year MATLAB Project - Enhanced Version
% Dataset: 1000 cats + 1000 dogs images
% Split: 80% Training, 20% Testing with validation split

clear all;
close all;
clc;

%% 1. LOAD AND PREPARE DATASET
fprintf('=== ADVANCED CAT vs DOG CLASSIFICATION PROJECT ===\n\n');
fprintf('Step 1: Loading dataset...\n');

% Define dataset path
datasetPath = 'Dog_vs_Cat_Dataset';

% Create ImageDatastore
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display dataset info
fprintf('Total images: %d\n', numel(imds.Files));
labelCount = countEachLabel(imds);
disp(labelCount);

% Check image sizes
fprintf('\nAnalyzing image dimensions...\n');
sampleImg = readimage(imds, 1);
fprintf('Sample image size: %dx%dx%d\n', size(sampleImg, 1), size(sampleImg, 2), size(sampleImg, 3));

%% 2. VISUALIZE SAMPLE IMAGES
fprintf('\nStep 2: Creating sample visualization...\n');

figure('Name', 'Sample Images from Dataset', 'Position', [100 100 1200 700]);
numSamples = 16;
for i = 1:numSamples
    subplot(4, 4, i);
    idx = randi(numel(imds.Files));
    img = readimage(imds, idx);
    imshow(img);
    title(sprintf('%s', char(imds.Labels(idx))), 'FontSize', 10, 'FontWeight', 'bold');
end
sgtitle('Sample Images: Cats and Dogs Dataset', 'FontSize', 16, 'FontWeight', 'bold');

%% 3. SPLIT DATA: 70% TRAINING, 15% VALIDATION, 15% TESTING
fprintf('\nStep 3: Splitting data (70%% train, 15%% validation, 15%% test)...\n');

% Split for training and temp (val + test)
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.7, 'randomized');

% Split temp into validation and test
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp, 0.5, 'randomized');

fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n', numel(imdsValidation.Files));
fprintf('Testing images: %d\n', numel(imdsTest.Files));

% Count labels
trainLabelCount = countEachLabel(imdsTrain);
valLabelCount = countEachLabel(imdsValidation);
testLabelCount = countEachLabel(imdsTest);

% Plot data split
figure('Name', 'Data Split Visualization', 'Position', [100 100 1200 400]);
subplot(1, 3, 1);
bar(categorical({'Cat', 'Dog'}), trainLabelCount.Count);
title('Training Set (70%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Number of Images');
xlabel('Class');
grid on;
ylim([0 max(trainLabelCount.Count) + 50]);
text(1:2, trainLabelCount.Count, num2str(trainLabelCount.Count), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 11);

subplot(1, 3, 2);
bar(categorical({'Cat', 'Dog'}), valLabelCount.Count);
title('Validation Set (15%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Number of Images');
xlabel('Class');
grid on;
ylim([0 max(trainLabelCount.Count) + 50]);
text(1:2, valLabelCount.Count, num2str(valLabelCount.Count), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 11);

subplot(1, 3, 3);
bar(categorical({'Cat', 'Dog'}), testLabelCount.Count);
title('Testing Set (15%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Number of Images');
xlabel('Class');
grid on;
ylim([0 max(trainLabelCount.Count) + 50]);
text(1:2, testLabelCount.Count, num2str(testLabelCount.Count), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 11);

sgtitle('Dataset Distribution', 'FontSize', 14, 'FontWeight', 'bold');

%% 4. LOAD PRE-TRAINED NETWORK (ResNet-18 - Better than GoogLeNet)
fprintf('\nStep 4: Loading pre-trained network (ResNet-18)...\n');

% Load ResNet-18
net = resnet18;
inputSize = net.Layers(1).InputSize;
fprintf('Network: ResNet-18\n');
fprintf('Input size required: %dx%dx%d\n', inputSize(1), inputSize(2), inputSize(3));

%% 5. DATA AUGMENTATION FOR BETTER TRAINING
fprintf('\nStep 5: Setting up data augmentation...\n');

% Training augmentation (helps prevent overfitting)
augmenterTrain = imageDataAugmenter(...
    'RandRotation', [-20, 20], ...
    'RandXReflection', true, ...
    'RandYReflection', false, ...
    'RandXScale', [0.8 1.2], ...
    'RandYScale', [0.8 1.2]);

% No augmentation for validation and test
augmenterValTest = imageDataAugmenter();

% Create augmented image datastores
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenterTrain, 'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'DataAugmentation', augmenterValTest, 'ColorPreprocessing', 'gray2rgb');

augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'DataAugmentation', augmenterValTest, 'ColorPreprocessing', 'gray2rgb');

fprintf('Data augmentation applied: Rotation, Flipping, Scaling\n');

%% 6. MODIFY NETWORK FOR TRANSFER LEARNING
fprintf('\nStep 6: Modifying network for binary classification...\n');

% Extract layer graph
lgraph = layerGraph(net);

% Get number of classes
numClasses = numel(categories(imdsTrain.Labels));
fprintf('Number of classes: %d\n', numClasses);

% Replace final layers
newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);

newClassificationLayer = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassificationLayer);

%% 7. TRAIN THE NETWORK
fprintf('\nStep 7: Training the network...\n');
fprintf('This will take several minutes. Training in progress...\n\n');

% Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'ValidationPatience', 5, ...
    'L2Regularization', 0.0001);

% Train network
tic;
trainedNet = trainNetwork(augimdsTrain, lgraph, options);
trainingTime = toc;

fprintf('\nTraining completed in %.2f seconds (%.2f minutes)\n', trainingTime, trainingTime/60);

%% 8. EVALUATE ON VALIDATION SET
fprintf('\nStep 8: Evaluating on validation set...\n');

[valPredictions, valScores] = classify(trainedNet, augimdsValidation);
valLabels = imdsValidation.Labels;
valAccuracy = mean(valPredictions == valLabels) * 100;

fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);

%% 9. EVALUATE ON TEST SET
fprintf('\nStep 9: Evaluating on test set...\n');

tic;
[testPredictions, testScores] = classify(trainedNet, augimdsTest);
testTime = toc;

testLabels = imdsTest.Labels;
testAccuracy = mean(testPredictions == testLabels) * 100;

fprintf('Test Accuracy: %.2f%%\n', testAccuracy);
fprintf('Prediction time: %.2f seconds\n', testTime);

%% 10. CONFUSION MATRIX
fprintf('\nStep 10: Generating confusion matrix...\n');

confMat = confusionmat(testLabels, testPredictions);
fprintf('Confusion Matrix:\n');
disp(confMat);

% Plot confusion matrix
figure('Name', 'Confusion Matrix', 'Position', [100 100 700 600]);
cm = confusionchart(testLabels, testPredictions);
cm.Title = sprintf('Test Set Confusion Matrix (Accuracy: %.2f%%)', testAccuracy);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% 11. CALCULATE DETAILED METRICS
fprintf('\nStep 11: Calculating performance metrics...\n');

classes = categories(testLabels);
numClasses = length(classes);

precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1score = zeros(numClasses, 1);
specificity = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;
    
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    specificity(i) = TN / (TN + FP);
end

% Create results table
resultsTable = table(classes, precision * 100, recall * 100, f1score * 100, specificity * 100, ...
    'VariableNames', {'Class', 'Precision_%', 'Recall_%', 'F1Score_%', 'Specificity_%'});

fprintf('\n=== PERFORMANCE METRICS TABLE ===\n');
disp(resultsTable);

%% 12. VISUALIZATION: PERFORMANCE METRICS
figure('Name', 'Performance Metrics', 'Position', [100 100 1200 500]);

subplot(1, 4, 1);
bar(categorical(classes), precision * 100, 'FaceColor', [0.2 0.6 0.8]);
title('Precision', 'FontWeight', 'bold');
ylabel('Percentage (%)');
ylim([0 100]);
grid on;
for i = 1:length(classes)
    text(i, precision(i)*100 + 2, sprintf('%.1f%%', precision(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

subplot(1, 4, 2);
bar(categorical(classes), recall * 100, 'FaceColor', [0.8 0.4 0.2]);
title('Recall (Sensitivity)', 'FontWeight', 'bold');
ylabel('Percentage (%)');
ylim([0 100]);
grid on;
for i = 1:length(classes)
    text(i, recall(i)*100 + 2, sprintf('%.1f%%', recall(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

subplot(1, 4, 3);
bar(categorical(classes), f1score * 100, 'FaceColor', [0.4 0.8 0.4]);
title('F1-Score', 'FontWeight', 'bold');
ylabel('Percentage (%)');
ylim([0 100]);
grid on;
for i = 1:length(classes)
    text(i, f1score(i)*100 + 2, sprintf('%.1f%%', f1score(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

subplot(1, 4, 4);
bar(categorical(classes), specificity * 100, 'FaceColor', [0.8 0.6 0.2]);
title('Specificity', 'FontWeight', 'bold');
ylabel('Percentage (%)');
ylim([0 100]);
grid on;
for i = 1:length(classes)
    text(i, specificity(i)*100 + 2, sprintf('%.1f%%', specificity(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

sgtitle('Classification Performance Metrics', 'FontSize', 16, 'FontWeight', 'bold');

%% 13. ROC CURVE AND AUC
fprintf('\nStep 12: Generating ROC curve...\n');

% Get scores for positive class (dog)
dogIdx = find(strcmp(classes, 'dog'));
positiveClassScores = testScores(:, dogIdx);
binaryLabels = double(testLabels == 'dog');

[X, Y, T, AUC] = perfcurve(binaryLabels, positiveClassScores, 1);

figure('Name', 'ROC Curve', 'Position', [100 100 700 600]);
plot(X, Y, 'LineWidth', 2.5, 'Color', [0.2 0.4 0.8]);
xlabel('False Positive Rate', 'FontSize', 12);
ylabel('True Positive Rate', 'FontSize', 12);
title(sprintf('ROC Curve (AUC = %.4f)', AUC), 'FontSize', 14, 'FontWeight', 'bold');
grid on;
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
legend({'ROC Curve', 'Random Classifier'}, 'Location', 'southeast', 'FontSize', 11);
hold off;

fprintf('AUC (Area Under Curve): %.4f\n', AUC);

%% 14. VISUALIZE SAMPLE PREDICTIONS
fprintf('\nStep 13: Creating prediction visualization...\n');

figure('Name', 'Sample Test Predictions', 'Position', [100 100 1400 800]);
numTestSamples = 20;
testIndices = randperm(numel(imdsTest.Files), numTestSamples);

correctCount = 0;
wrongCount = 0;

for i = 1:numTestSamples
    subplot(4, 5, i);
    idx = testIndices(i);
    img = readimage(imdsTest, idx);
    imshow(img);
    
    actualLabel = char(testLabels(idx));
    predictedLabel = char(testPredictions(idx));
    confidence = max(testScores(idx, :)) * 100;
    
    if strcmp(actualLabel, predictedLabel)
        titleColor = [0 0.6 0];
        resultText = '✓ CORRECT';
        correctCount = correctCount + 1;
    else
        titleColor = [0.9 0 0];
        resultText = '✗ WRONG';
        wrongCount = wrongCount + 1;
    end
    
    title(sprintf('%s (%.1f%%)\nTrue: %s | Pred: %s', resultText, confidence, actualLabel, predictedLabel), ...
        'Color', titleColor, 'FontSize', 9, 'FontWeight', 'bold');
end
sgtitle(sprintf('Sample Predictions (Correct: %d, Wrong: %d)', correctCount, wrongCount), ...
    'FontSize', 16, 'FontWeight', 'bold');

%% 15. MISCLASSIFIED IMAGES ANALYSIS
fprintf('\nStep 14: Analyzing misclassified images...\n');

misclassifiedIdx = find(testPredictions ~= testLabels);
numMisclassified = length(misclassifiedIdx);

fprintf('Total misclassified images: %d out of %d\n', numMisclassified, numel(testLabels));

if numMisclassified > 0
    figure('Name', 'Misclassified Images', 'Position', [100 100 1200 700]);
    numShow = min(12, numMisclassified);
    
    for i = 1:numShow
        subplot(3, 4, i);
        idx = misclassifiedIdx(i);
        img = readimage(imdsTest, idx);
        imshow(img);
        
        actualLabel = char(testLabels(idx));
        predictedLabel = char(testPredictions(idx));
        confidence = max(testScores(idx, :)) * 100;
        
        title(sprintf('True: %s | Pred: %s (%.1f%%)', actualLabel, predictedLabel, confidence), ...
            'Color', [0.9 0 0], 'FontSize', 9, 'FontWeight', 'bold');
    end
    sgtitle('Misclassified Images Analysis', 'FontSize', 16, 'FontWeight', 'bold');
end

%% 16. CONFIDENCE DISTRIBUTION
fprintf('\nStep 15: Analyzing confidence distribution...\n');

maxConfidences = max(testScores, [], 2) * 100;

figure('Name', 'Confidence Distribution', 'Position', [100 100 1000 500]);

subplot(1, 2, 1);
histogram(maxConfidences, 20, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'black');
xlabel('Confidence (%)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Overall Confidence Distribution', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
xline(mean(maxConfidences), 'r--', sprintf('Mean: %.1f%%', mean(maxConfidences)), ...
    'LineWidth', 2, 'FontSize', 11);

subplot(1, 2, 2);
correctIdx = testPredictions == testLabels;
histogram(maxConfidences(correctIdx), 20, 'FaceColor', [0.2 0.8 0.3], 'EdgeColor', 'black');
hold on;
histogram(maxConfidences(~correctIdx), 20, 'FaceColor', [0.9 0.2 0.2], 'EdgeColor', 'black');
xlabel('Confidence (%)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Confidence: Correct vs Incorrect', 'FontSize', 13, 'FontWeight', 'bold');
legend({'Correct Predictions', 'Incorrect Predictions'}, 'FontSize', 10);
grid on;
hold off;

%% 17. TRAINING SUMMARY TABLE
summaryData = {
    'Total Images', numel(imds.Files);
    'Training Images', numel(imdsTrain.Files);
    'Validation Images', numel(imdsValidation.Files);
    'Testing Images', numel(imdsTest.Files);
    'Network Architecture', 'ResNet-18 (Transfer Learning)';
    'Input Size', sprintf('%dx%d', inputSize(1), inputSize(2));
    'Data Augmentation', 'Yes (Rotation, Flip, Scale)';
    'Epochs', '15';
    'Mini-batch Size', '32';
    'Validation Accuracy (%)', sprintf('%.2f', valAccuracy);
    'Test Accuracy (%)', sprintf('%.2f', testAccuracy);
    'AUC Score', sprintf('%.4f', AUC);
    'Training Time (min)', sprintf('%.2f', trainingTime/60);
    'Prediction Time (sec)', sprintf('%.2f', testTime);
    'Misclassified Images', numMisclassified
};

summaryTable = cell2table(summaryData, 'VariableNames', {'Metric', 'Value'});
fprintf('\n=== PROJECT SUMMARY ===\n');
disp(summaryTable);

%% 18. CLASS-WISE ACCURACY
classAccuracy = zeros(numClasses, 1);
for i = 1:numClasses
    classIdx = testLabels == classes{i};
    classAccuracy(i) = mean(testPredictions(classIdx) == testLabels(classIdx)) * 100;
end

classAccuracyTable = table(classes, classAccuracy, 'VariableNames', {'Class', 'Accuracy_%'});
fprintf('\n=== CLASS-WISE ACCURACY ===\n');
disp(classAccuracyTable);

figure('Name', 'Class-wise Accuracy', 'Position', [100 100 600 500]);
bar(categorical(classes), classAccuracy, 'FaceColor', [0.4 0.5 0.8]);
ylabel('Accuracy (%)', 'FontSize', 12);
xlabel('Class', 'FontSize', 12);
title('Per-Class Test Accuracy', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0 100]);
grid on;
for i = 1:length(classes)
    text(i, classAccuracy(i) + 2, sprintf('%.2f%%', classAccuracy(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

%% 19. SAVE MODEL AND RESULTS
fprintf('\nStep 16: Saving trained model and results...\n');

save('advanced_cat_dog_model.mat', 'trainedNet', 'inputSize', 'classes', ...
    'testAccuracy', 'valAccuracy', 'confMat', 'resultsTable', 'summaryTable', ...
    'AUC', 'classAccuracyTable', '-v7.3');

fprintf('Model saved as: advanced_cat_dog_model.mat\n');
