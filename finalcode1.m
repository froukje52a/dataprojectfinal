%% Project Code

%% info
%Monkey ID: H

%kinematics
%Sampling frequencies (during the experiment): 800 Hz
%Smoothing: Gaussian kernel with a standard deviation of 20 ms 
%Binning: 9.8 ms

%Variables (matrices in the dataset)
%hand_pos —> positions in x and y
%hand_vel —> velocities in x and y
%Matrix dimensions: Time bin x kinematic dimension (x,y)

%Neural recording
%Sampling frequencies (during the experiment): 800 Hz
%Smoothing: Gaussian kernel with a standard deviation of 20 ms 
%Binning: 9.8 ms

%Variable (matrix in the dataset)
%f_rates —> firing rates per neuron
%Matrix dimension: Time bin x neuron

%% Load the data
BMI_data= load('BMI_Data_Froukje.mat');

%% preprocessing
%the firing rate 17097 bins with 113 neurons
f_rates= BMI_data.f_rates;
%hand position hand_pos Time bin x kinematic dimension (x,y): [17097×2 double]
hand_pos= BMI_data.hand_pos;
%hand velocity hand_vel Time bin x kinematic dimension (x,y)[17097×2 double]
hand_vel= BMI_data.hand_vel;

%preprocess f_rates
f_rates= zscore(f_rates);

%number of bins
bins = size(f_rates, 1); 

%make kinematics matrix
kinematics = [hand_pos, hand_vel];

%preprocess kinematics
kinematics= kinematics- mean(kinematics);

%set all the possible offsets of forward estimation and sensory feedback
offsets = [-18,-17,-16,-15,-14,-13,-12, -11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11];
%[1] = (-18, -176.4 to -166.6) [2] = (-17, -166.6 to -156.8) [3] = (-16, -156.8 to -147.0) 
%[4] = (-15, -147.0 to -137.2) [5] = (-14, -137.2 to -127.4) [6] = (-13, -127.4 to -117.6)
%[7] = (-12, -117.6 to -107.8) [8] = (-11, -107.8 to -98.0)  [9] = (-10, -98.0 to -88.2)
%[10] = (-9, -88.2 to -78.4)  [11] = (-8, -78.4 to -68.6) [12] = (-7, -68.6 to -58.8
%[13] = (-6, -58.8 to -49.0)  [14] = (-5, -49.0 to -39.2) [15] = (-4, -39.2 to -29.4)
%[16] = (-3, -29.4 to -19.6)  [17] = (-2, -19.6 to -9.8)  [18] = (-1, -9.8 to 0)
%[19] = (0, 0 to 9.8)         [20] = (1, 9.8 to 19.6)     [21] = (2, 19.6 to 29.4)
%[22] = (3, 29.4 to 39.2)     [23] = (4, 39.2 to 49.0)    [24] = (5, 49.0 to 58.8)
%[25] = (6, 58.8 to 68.6)     [26] = (7, 68.6 to 78.4)    [27] = (8, 78.4 to 88.2)
%[28] = (9, 88.2 to 98.0)     [29] = (10, 98.0 to 107.8)  [30] = (11, 107.8 to end)

time_offsets_start= [-176.4, -166.6, -156.8, -147.0, -137.2, -127.4, -117.6, -107.8, -98.0, -88.2, -78.4, -68.6, -58.8, -49.0, -39.2, -29.4, -19.6, -9.8, 0, 9.8, 19.6, 29.4, 39.2, 49.0, 58.8, 68.6, 78.4, 88.2, 98.0, 107.8];
%% load preprocess the S1_unit_guide
%because of large file in github I already extractecd S1_unit_guide
%S1_unit_guide= trial_data.S1_unit_guide;  save('S1_unit_guide.mat', 'S1_unit_guide');

%load the the S1_unit guide
load('S1_unit_guide.mat')
S1_unit_guide= double(S1_unit_guide);%make it a double 
%make the index starting from 1 instead of 0 for each neuron
S1_unit_guide(:,2) = S1_unit_guide(:,2) + 1;

electrode = readtable('elec_map.csv');%Load the electrodes
save('electrode.mat', 'electrode');

%only get the "interesting" arrays of the table
electrodes = table2array(electrode(:, {'chan', 'rowNum', 'colNum'}));
%matrix with column 1 all the channels, column 2 the rowNumbers and columnnumber

%cross-validation on 10 folds
K_fold = 10;

%initiliase constraint, because of ridge they are all []
A = [];
b = [];
Aeq = [];
beq = [];

%bounds of lambda
lb=0;
ub=1;

%initiliase options
options = optimoptions(@fmincon,'Display','off');

%intiliase lambdas and MSE
lambdas= zeros(K_fold, length(offsets));
each_MSE= zeros(K_fold, length(offsets));

%initilaise allinds
allInds = 1:K_fold;

%intialise p_values and R_squared
p_values = zeros(K_fold,length(offsets));
R_squared = zeros(K_fold, length(offsets));

%number of r squared values per fold
number = 10;

%initialise R_all, all the R_squared values of for each fold*number for
%each offset and each neuron
R_all= zeros(number*K_fold, length(offsets), 113);

%% perform the ridge regression for the different offsets and folds
%do it number(10) of times
for c= 1:number
    %loop over each fold crossvalidation
    for neuron_idx  = 1:K_fold  
        %go over each offset value
        for t=1:length(offsets)
            %start with the initial kinematics
            kinematics_offset= kinematics;
            f_rates_offset= f_rates;
            %forward estimation values
            if offsets(t) < 0
              kinematics_offset= kinematics_offset(1:(end+offsets(t)), :);
              f_rates_offset= f_rates_offset(1-offsets(t):end, :);
            %sensory feedback values
            elseif offsets(t) > 0
            kinematics_offset=kinematics_offset(offsets(t)+1:end, :);
            f_rates_offset= f_rates_offset(1:(end-offsets(t)), :);
            %if the kinematics are null offset
            else
                kinematics_offset= kinematics;
                f_rates_offset= f_rates;
            end
    
        %needs to change for every offset
        n = size(kinematics_offset, 1);
        indices = crossvalind('Kfold', n, K_fold);
    
        %split data into train, test and validation indices
        trainInds = allInds(allInds~=neuron_idx);
        validInds = max(trainInds);
        train = ismember(indices, trainInds(trainInds~=validInds));
        validation = ismember(indices, validInds); 
        test = (indices == neuron_idx); 
   
        %split data into train, test and validation
        train_kinematics = kinematics_offset(train, :);
        train_f_rates =  f_rates_offset(train, :);
        validation_kinematics = kinematics_offset(validation, :);
        validation_f_rates =  f_rates_offset(validation, :);
        test_kinematics = kinematics_offset(test, :);
        test_f_rates =  f_rates_offset(test, :);

        %use ridge regression function and fmincon to find optimal values for
        %lambda
        %for j = 1:length(lambda)
        MSE = @(lambda) ridge_regression(train_kinematics, train_f_rates, validation_kinematics, validation_f_rates, lambda);
        [optimal_lambda, fval] = fmincon(MSE,0, A, b, Aeq, beq, lb, ub, [], options);

        %store the optimal lambdas for each fold and every offset
        lambdas(neuron_idx,t) = optimal_lambda;
        
        %find optimal weights and minimum of squared errors for the lambda
        [lambda_mse, B_weights]= ridge_regression(train_kinematics, train_f_rates, test_kinematics, test_f_rates, optimal_lambda);
        %predict firing rate
        test_pred = test_kinematics * B_weights;
        
        %calculate for each fold and offset
        for neurons_channel = 1:size(f_rates_offset, 2)
            [R,p] = corrcoef(test_pred(:,neurons_channel), test_f_rates(:,neurons_channel));
            R_squared(neuron_idx,t,neurons_channel) = R(1,2)^2;
            p_values(neuron_idx,t, neurons_channel) = p(2,1);
        end
   
        %store mse for optimal lambdas
        each_MSE(neuron_idx,t) = ridge_regression(train_kinematics, train_f_rates, validation_kinematics, validation_f_rates, optimal_lambda);
        
        %get the correct value assigned to the R_all
        R_all((c-1)*10+1:c*10, :,:) = R_squared;
        end      
    end
end
%% analyse the R_squared

%mean MSE
mean_mse= mean(each_MSE);

%remove the first unnecessary dimension
r2_mean_all = squeeze(mean(R_all, 1));

%null offset R_squared 
R2_mean_0offset= r2_mean_all(19,:);
%mean value over all 113 neurons
value_R2_mean_0offset= mean(R2_mean_0offset);

figure(1)
plot(R2_mean_0offset)
xlabel('neurons')
ylabel('R squared values')
title('maximum R squared for each neuron in null offset')

%mean value per offset
r2_mean_all_offset= mean(r2_mean_all,2);

figure(2)
%shows the mean R_squared per offset index 
plot(time_offsets_start, r2_mean_all_offset);
title('mean R squared  per offset');
xlabel('time offset start');
ylabel('mean R^2 per per offset');

%Wilcoxon signed-rank test comparing each column to the 0 value
p_values_wilcoxon = zeros(30,1);
for col_offset = 1:30
        [p_values_wilcoxon(col_offset), ~] = signrank(r2_mean_all(col_offset,:), r2_mean_all(19,:));
end
p_values_log = -log10(p_values_wilcoxon);

figure(3)
%shows the wilcoxon signed rank test, comparing all the indexes maximum 
plot(time_offsets_start, p_values_log);
hold on;
%plot the significance line
significance= -log10(0.05);
yline(significance)
hold off;
title('-log10 of the p_values of the time offset against null shift, including null shift');
xlabel('time offset start');
ylabel('-log10 p_values against null shift' );
legend('-log10 of the p values', '-log10(0.05)')

%% make table into matrix with the channels in the correct place for the matrix

%only get the information of our monkey H, 113 neurons
electrodes_h= electrodes(97:192, :);%monkey H is from 97-192

%generate the matrix_location with its correct placement of the channels
[loc_channel, matrix_location]= generate_matrix_location(f_rates, electrodes_h, S1_unit_guide);

%% line them up correctly the maximum R squared of the mean values with the maximum index
%find the maximum of each neuron and its corresponding index
[maximum_neuron, index ] = max(r2_mean_all);

%preprocess the maximum neuron and R_squared for the location matrixes
index= index(:, 1:end)';
maximum_neuron= maximum_neuron(:, 1:end)';
%concatenate maximum_neuron and index into max_index_and_neuron
max_index_and_neuron= horzcat(index, maximum_neuron);

%concatenate the max_index_and_neuron with S1_unit_guide
matrix_info= horzcat(max_index_and_neuron, S1_unit_guide);
%remove the fourth column because not of interest here
matrix_info= matrix_info(:, 1:3);

%initiliase an array with the maximum offset
max_offset = zeros(96, 1);

%keeps only the corrsponding index of the maximum value of mean values of the R2
%use accumarray to group the first and third collumn and find its max between them 
grouped_vals_offset = accumarray(matrix_info(:, 3), matrix_info(:, 1), [], @max);
%set the maximum value of the grouped_vals_offset to each row coressponding to the channel
max_offset(1:size(grouped_vals_offset, 1)) = grouped_vals_offset; 

all_rows_offset = size(max_offset, 1);
all_indices_offset = (1:all_rows_offset);
%concatenate the channel with the max_offset
max_offset= horzcat(all_indices_offset', max_offset);

%% the maximum index at the correct location matrix
%initialise index matrix
index_matrix = zeros(size(matrix_location));

%loop over the rows of the matrix_location
for rows = 1:size(matrix_location, 1)
    %loop over the columns of the matrix_location
    for cols = 1:size(matrix_location, 2)
        %compare the channel of max_offset with the matrix location and obtain 
        %the value of that channel out of the second column of max_offset.
        index_offset = max_offset(max_offset(:, 1) == matrix_location(rows, cols), 2);
        %account for empty values 
        if isempty(index_offset)
            index_matrix(rows, cols) = 0;
        else
            %assign the maximum value of the neuron to index_matrix
            index_matrix(rows, cols) = index_offset;
        end
    end
end

%% initiliase an array with maximum value for the each neuron or the maximum offset
max_neuron_i = zeros(96, 1);
%initialise R_squared_matrix
R_squared_matrix1 = zeros(size(matrix_location));

R_all_columns= mean(R_all,1);
for i= 1:30
    [maximum_neuron_index, index ] = max(R_all_columns(:,i,:), [], 2);

    %preprocess R_squared for the location matrixes over different time points
    maximum_neuron_i= maximum_neuron_index(:, 1:end)';

    %concatenate with S1_unit_guide
    matrix_info_i= horzcat(maximum_neuron_i, S1_unit_guide);
    %remove the third column
    matrix_info_i= matrix_info_i(:, 1:2);

    %find its max between them
    group = accumarray(matrix_info_i(:,2), matrix_info_i(:, 1), [], @max);
    max_neuron_i(1:size(group, 1)) = group; 

    %assign the channels to the values
    all_rows = size(max_neuron_i, 1);
    all_indices = (1:all_rows);
    %concatenate the channel with the max_neuron
    max_neuron_i= horzcat(all_indices', max_neuron_i);

    %loop over the rows of the matrix_location
    for rows = 1:size(matrix_location, 1)
        %loop over the columns of the matrix_location
        for cols = 1:size(matrix_location, 2)
            %compare the channel of max_neuron with the matrix location and obtain 
            %the value of that channel out of the second column of max_neuron.
            maximumvalue_i = max_neuron_i(max_neuron_i(:, 1) == matrix_location(rows, cols), 2);
            %account for empty values 
            if isempty(maximumvalue_i)
                R_squared_matrix1(rows, cols) = 0; 
            else
                %assign the maximum value of the neuron to the the R_squared
                R_squared_matrix1(rows, cols) = maximumvalue_i;
            end
        end
    end

figure(5)
clear_R_squared_offset = R_squared_matrix1;
clear_R_squared_offset(clear_R_squared_offset >.3) = 0.3;
subplot(5, 6, i);
imagesc(clear_R_squared_offset )
clear_R_squared_offset (isnan(clear_R_squared_offset ))=0;
colors = colormap;
%set nan value a different color
colors(1, :) =  [0.5 0.5 0.5]; 
colormap(colors);
colorbar;
imagesc(clear_R_squared_offset )
title(time_offsets_start(i))

figure(4)
if i==19
    clear_R_squared = R_squared_matrix1;
    clear_R_squared (clear_R_squared >.3) = 0.3;
    imagesc(clear_R_squared )
    clear_R_squared (isnan(clear_R_squared ))=0;
    imagesc(clear_R_squared )
    colormap(colors);
    colorbar;
    title('R squared in null offset ')
    xlabel('columns')
    ylabel('rows')
end 
end 

%% mean overall performance at different time points
%taking the mean of each column
index_matrix(index_matrix == 0) = NaN;

figure(6);
index_visual_matrix = zeros(10,10);
index_visual_matrix(isnan(index_matrix)) = 0;
%sensory feedback
index_visual_matrix(index_matrix < 16) = 1;
%forward estimation
index_visual_matrix(index_matrix >= 16 ) = 2; 
colormap(colors);
imagesc(index_visual_matrix);
title('<= -29.4 sensory feedback')
xlabel('columns')
ylabel('rows')

%% plot the maximum offset index and its corresponding maximum R squared
index_time_offset =(1:30)';
max_index_and_neuron1= max_index_and_neuron;

%get starting value 
sameindex = time_offsets_start(index_time_offset);
for i = 1:size(max_index_and_neuron1, 1)
    if max_index_and_neuron1(i, 1) <= numel(sameindex)
       max_index_and_neuron1(i, 1) = sameindex(max_index_and_neuron1(i, 1));
    end
end

%only get the unique ones
time_offset = unique(max_index_and_neuron1(:, 1)); 
%intiliase 
maximum_r_at_offset = zeros(size(time_offset));  

for i = 1:length(time_offset)
    %take the mean for multiple values of the same offset
    maximum_r_at_offset(i) = mean(max_index_and_neuron1(max_index_and_neuron1(:, 1) == time_offset(i), 2)); 
end

figure(7)
bar(time_offset, maximum_r_at_offset);
xlabel("time offset start")
ylabel('R2 values')
title("max R2 at the time offset start")

%% plot the number of maximum at that offset index of the max r squared 
figure(8)
number_val = max_index_and_neuron1(:, 1);  

hist_bar = histogram(number_val, time_offsets_start);
count = hist_bar.BinCounts;
edge = hist_bar.BinEdges;
offset = (edge(1:end-1) + edge(2:end)) / 2;
xlabel('time offset start');
ylabel('count');
title('number of neurons in each time offset bin')

%% Functions

function [lambda_mse, B_weights]= ridge_regression(X_train, Y_train,X_validation, Y_validation, lambda)
%[lambda_mse, B_weights]= ridge_regression(X_train, Y_train,X_validation, Y_validation, lambda)
%input: X_train:training x value,independent(kinematics)
%       Y_train:training y value,dependent(f_rate)
%       X_validation: validation x value, independent(kinematics)
%       Y_validation: validation y value, dependent(f_rated)
%       lambda: is the given lambda value
%output: lambda_mse: means squared error for each lambda value
%        B_weights: the weights for each beta value for the ridge regression 
%the function performs ridge regression and finds the mse and beta weights 
%for the given X_train,Y_train,X_validation, Y_validation, lambda values

%initiliase parts of the ridge regression formula (X^T X + λI)^-1 X^T y
%eye is identity I
I = eye(size(X_train, 2));
%X^T X 
X_tX = X_train.' * X_train;
%λI
lambda_identity = lambda * I;
%implement ridge regression
B_weights = (X_tX + lambda_identity) \ (X_train.' * Y_train);

Y_hat = X_validation*B_weights;
lambda_mse= sum(mean(Y_hat-Y_validation)).^2;
end

function [loc_channel, matrix_location]= generate_matrix_location(f_rates, electrodes_h, S1_unit_guide)
%%[loc_channel, matrix_location]= generate_matrix_location(f_rates, electrodes_h, S1_unit_guide)
%input: loc_channel:firing rates for the neurons
%       electrodes_h: electrodes of monkey h, its channel,row,column
%       S1_unit_guide: validation y value, dependent(f_rated)
%output: loc_channel: channel with its corresponding row and channel
%        matrix_location: the channel nummber on the right location
%the function uses the channel and its corresponding row and column of 
%monkey h to make the loc_channel matrix for each electrode. and it uses
%the loc_channel matrix gto make the matrix_location matrix, to get each
%channel at the right spot.

%initialise the channels matrix which contains the channel and row and
%column for every electrode
loc_channel = zeros(size(f_rates,2),3);

%initialise the base matrix_location matrix
matrix_location= zeros(10,10);

%loop over every channel
for ch = 1:size(electrodes_h, 1)
    %obtain the channel
    channel = electrodes_h(ch, 1);
    %find the indexes of neurons which have the same channel
    chan_index = find(S1_unit_guide(:,1) == channel);
    
    %loop over set of indexes
    for neuron = 1:length(chan_index)
       %only use the second and third row to put in the neurons at the
       %correct spot for each of the matching channels of neuron
       loc_channel(chan_index(neuron), :) = [channel, electrodes_h(ch, 2:3)];
    end  
end

%go over the loc_channel matrix and assign the correct row and column to
%the channel number
for neuron_idx = 1:size(loc_channel, 1)
    %set the channel, row, col to the correct place
    chan= loc_channel(neuron_idx, 1);
    row=  loc_channel(neuron_idx, 2);
    col=  loc_channel(neuron_idx, 3);
    matrix_location(row, col) = chan;
end
end 