%% Project

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

%set all the possible shifts of the efference copy and sensory feedback
shifts = [-18,-17,-16,-15,-14,-13,-12, -11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11];
% -18:-176.4,-17: -166.6,-16: -156.8,-15: -147,-14:-137.2, -13:127.4,
% -12:-117.6, -11:-107.8,-10:-98, -9:-88.2,-8: -78.4, -7:-68.6,-6: -58.8, 
% -5:-49, -4: -39.2, -3: -29,4, -2: -19.6, -1:-9.8, 0:0
% 1:9.8, 2:19.6, 3:29.4, 4: 39.2, 5:49.0, 6:58.8, 7:68.6, 8:78.4, 9:88.2
% 10:98.0, 11: 107.8

%1: -176.4, 2: -166.6, 3: -156.8, 4: -147, 5: -137.2, 6: 127.4, 
%7: -117.6, 8: -107.8, 9: -98, 10: -88.2, 11: -78.4, 12: -68.6, 13: -58.8, 
%14: -49, 15: -39.2, 16: -29.4, 17: -19.6, 18: -9.8, 19: 0, 
%20: 9.8, 21: 19.6, 22: 29.4, 23: 39.2, 24: 49.0, 25: 58.8, 26: 68.6, 
%27: 78.4, 28: 88.2, 29: 98.0, 30: 107.8

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
lambdas= zeros(K_fold, length(shifts));
each_MSE= zeros(K_fold, length(shifts));

%initilaise allinds
allInds = 1:K_fold;

%intialise p_values and R_squared
p_values = zeros(K_fold,length(shifts));
R_squared = zeros(K_fold, length(shifts));

%number of r squared values per fold
number = 10;

%initialise R_all, all the R_squared values of for each fold*number for
%each shift and each neuron
R_all= zeros(number*K_fold, length(shifts), 113);


%% perform the ridge regression for the different time shifts and folds
%do it number(10) of times
for c= 1:number
    %loop over each fold crossvalidation
    for neuron_idx  = 1:K_fold  
        %go over each shift
        for t=1:length(shifts)
            %start with the initial kinematics
            kinematics_shift= kinematics;
            f_rates_shift= f_rates;
            %for the efference values
            if shifts(t) < 0
              kinematics_shift= kinematics_shift(1:(end+shifts(t)), :);
              f_rates_shift= f_rates_shift(1-shifts(t):end, :);
            %for the sensory feedback
            elseif shifts(t) > 0
            kinematics_shift=kinematics_shift(shifts(t)+1:end, :);
            f_rates_shift= f_rates_shift(1:(end-shifts(t)), :);
            %if the kinematics are not shifted
            else
                kinematics_shift= kinematics;
                f_rates_shift= f_rates;
            end
    
        %needs to change for every shift
        n = size(kinematics_shift, 1);
        indices = crossvalind('Kfold', n, K_fold);
    
        %split data into train, test and validation indices
        trainInds = allInds(allInds~=neuron_idx);
        validInds = max(trainInds);
        train = ismember(indices, trainInds(trainInds~=validInds));
        validation = ismember(indices, validInds); 
        test = (indices == neuron_idx); 
   
        %split data into train, test and validation
        train_kinematics = kinematics_shift(train, :);
        train_f_rates =  f_rates_shift(train, :);
        validation_kinematics = kinematics_shift(validation, :);
        validation_f_rates =  f_rates_shift(validation, :);
        test_kinematics = kinematics_shift(test, :);
        test_f_rates =  f_rates_shift(test, :);

        %use ridge regression function and fmincon to find optimal values for
        %lambda
        %for j = 1:length(lambda)
        MSE = @(lambda) ridge_regression(train_kinematics, train_f_rates, validation_kinematics, validation_f_rates, lambda);
        [optimal_lambda, fval] = fmincon(MSE,0, A, b, Aeq, beq, lb, ub, [], options);

        %store the optimal lambdas for each fold and every shift
        lambdas(neuron_idx,t) = optimal_lambda;
        
        %find optimal weights and minimum of squared errors for the lambda
        [lambda_mse, B_weights]= ridge_regression(train_kinematics, train_f_rates, test_kinematics, test_f_rates, optimal_lambda);
        %predict firing rate
        test_pred = test_kinematics * B_weights;
        
        %calculate for each fold and shift
        for neurons_channel = 1:size(f_rates_shift, 2)
            [R,p] = corrcoef(test_pred(:,neurons_channel), test_f_rates(:,neurons_channel));
            R_squared(neuron_idx,t,neurons_channel) = R(1,2)^2;
            p_values(neuron_idx,t, neurons_channel) = p(2,1);
        end
        %[h0,p_t,c_I,test_stat] = ttest2(test_pred, test_f_rates);
   
        %store mse for optimal lambdas
        each_MSE(neuron_idx,t) = ridge_regression(train_kinematics, train_f_rates, validation_kinematics, validation_f_rates, optimal_lambda);
        
        %get the correct value assigned to the R_all
        R_all((c-1)*10+1:c*10, :,:) = R_squared;
        
        end   
        
    end
end
%% analyse the R_squared

%mean Mse
mean_mse= mean(each_MSE);

%null offset R_squared 
R2_0offset= R_squared(:,19,:);
%mean when in null offset
R2_mean_0offset= squeeze(mean(R2_0offset));
R2_mean_0offset_cutoff= R2_mean_0offset(R2_mean_0offset>0.05);
value_R2_mean_0offset= mean(R2_mean_0offset);
value_R2_mean_0offset_cutoff= mean(R2_mean_0offset_cutoff);
figure(1)
plot(R2_mean_0offset)
xlabel('neurons')
ylabel('R squared values')
title('maximum R squared for eache neuron aligned')
figure(2)
plot(R2_mean_0offset)
hold on;
plot(R2_mean_0offset_cutoff)
xlabel('neurons')
ylabel('R squared values')
title('maximum R squared for eache neuron aligned and aligned cut off')
legend('aligned', 'aligned cut off')


r2_mean_all = squeeze(mean(R_all, 1));
%significance?
%r2_mean_all = r2_mean_all(:,max(r2_mean_all)>.05);

r2_mean_all_rowindex= mean(r2_mean_all,2);

%figure
figure(3)
%shows maximum of the mean R_squared of all the rows which are significant
plot(r2_mean_all_rowindex);
title('maximum mean R squared of the signifcant elements per row');
xlabel('shift');
ylabel('mean R^2 per row');
%p_value_neuron= mean(p_values(:,:,mean)


%find the maximum of the mean
%R_all_columns= mean(R_all,1);
%R_all_columns= R_all_columns(R_all_columns>0.05);
[maximum_neuron, index ] = max(r2_mean_all, [], 2);
%[p,h,stats] = signrank(maximum_neuron,  R_all_columns(:,19,:));

%only look at significant values
R_all_sig= r2_mean_all(r2_mean_all(19,:)>0.05);
R_all_sig_at0= mean(R_all_sig);

%Wilcoxon signed-rank test comparing each column to the 0 value(even
%though it is efference copy)

p_values_wilcoxon = zeros(113,1);
for col_shift = 1:30
    %for neuron_coli= 1:113
        [p_values_wilcoxon(col_shift), ~] = signrank(r2_mean_all(col_shift,:), r2_mean_all(19,:));
    %end
end
p_values_log = -log10(p_values_wilcoxon);

figure(4)
%shows the wilcoxon signed rank test, comparing all the indexes maximum 
plot(p_values_log);
title('-log10 of the p_values of the index against the 0 index, including 0 itself');
xlabel('shift');
ylabel('-log10 p_values against the 0 index' );

%preprocess the maximum neuron and R_squared for the location matrixes
index= index(:, 1:end)';
maximum_neuron= maximum_neuron(:, 1:end)';
%the n inconsistent
%maximum_neuron= maximum_neuron(maximum_neuron>.05)
best_r= horzcat(index, maximum_neuron);

mean_index= mean(index);


%r2_mean_all = r2_mean_all(r2_mean_all(R_all, 1)>0.05);
%standard deviation
r2_std_all= std(maximum_neuron);
%standard error
r2_error = r2_std_all/sqrt(length(maximum_neuron));               
t_score = tinv([0.025  0.975],length(maximum_neuron)-1);

%% load preprocess the S1_unit_guide



%load the the S1_unit guide

%S1_unit_guide= trial_data.S1_unit_guide;  
%save('S1_unit_guide.mat', 'S1_unit_guide');
%make it a double 
load('S1_unit_guide.mat')
S1_unit_guide= double(S1_unit_guide);
%make the index starting from 1 instead of 0 for each neuron
S1_unit_guide(:,2) = S1_unit_guide(:,2) + 1;
unique_c = S1_unit_guide(:, 1); 
numUniqueNumbers = numel(unique(unique_c));
%% spatial analysation


%Load the electrodes
electrode = readtable('elec_map.csv');
save('electrode.mat', 'electrode');

%only get the "interesting" arrays of the matrix
electrodes = table2array(electrode(:, {'chan', 'rowNum', 'colNum'}));
%matrix with column 1 all the channels, column 2 the rowNumbers and column

%only get the informtaion of our monkey H, 113 neurons
electrodes_h= electrodes(97:192, :);%monkey H is from 97-192

%% make table into matrix with the channels in the correct place for the matrix

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

%figure(3)
%shows the loaction of the channels in the brain
disp(matrix_location)


%% line them up correctly the best_r with the 
%for i= 1:30
%    [maximum_neuron, index ] = max(R_all_columns(:,i,:), [], 2);
[maximum_neuron, index ] = max(r2_mean_all);

%preprocess the maximum neuron and R_squared for the location matrixes
index= index(:, 1:end)';
maximum_neuron= maximum_neuron(:, 1:end)';
%the n inconsistent
%maximum_neuron= maximum_neuron(maximum_neuron>.05)
best_r= horzcat(index, maximum_neuron);



%concatenatet the best_r with S1_unit_guide
matrix_info= horzcat(best_r, S1_unit_guide);
%remove the fourth column
matrix_info= matrix_info(:, 1:3);
%initiliase an array with maximum value for the each neuron or the maximum
%shift
max_neuron = zeros(96, 1);
max_shift = zeros(96, 1);


%use accumarray to group the second and third collumn and find its max
%between them
grouped_vals = accumarray(matrix_info(:, 3), matrix_info(:, 2), [], @max);
%set the maximum value of the grouped_vals to each row coressponding to the channel
max_neuron(1:size(grouped_vals, 1)) = grouped_vals; 

%use accumarray to group the first and third collumn and find its max between them 
grouped_vals_shift = accumarray(matrix_info(:, 3), matrix_info(:, 1), [], @max);
%set the maximum value of the grouped_vals_shift to each row coressponding to the channel
max_shift(1:size(grouped_vals_shift, 1)) = grouped_vals_shift; 



%%assign the channels to the values
all_rows = size(max_neuron, 1);
all_indices = (1:all_rows);
%concatenate the channel with the max_neuron
max_neuron= horzcat(all_indices', max_neuron);

all_rows_shift = size(max_shift, 1);
all_indices_shift = (1:all_rows_shift);
%concatenate the channel with the max_shift
max_shift= horzcat(all_indices_shift', max_shift);


%% display the maximum R_squared in the matrix and the maximum index according shift in the matrix
%initialise R_squared_matrix
R_squared_matrix = zeros(size(matrix_location));
%initialise index matrix
index_matrix = zeros(size(matrix_location));


%loop over the rows of the matrix_location
for rows = 1:size(matrix_location, 1)
    %loop over the columns of the matrix_location
    for cols = 1:size(matrix_location, 2)
        %compare the channel of max_neuron with the matrix location and obtain 
        %the value of that channel out of the second column of max_neuron.
        maximumvalue_neuron = max_neuron(max_neuron(:, 1) == matrix_location(rows, cols), 2);
        %compare the channel of max_shift with the matrix location and obtain 
        %the value of that channel out of the second column of max_shift.
        index_shift = max_shift(max_shift(:, 1) == matrix_location(rows, cols), 2);
        %account for empty values 
        if isempty(maximumvalue_neuron) || isempty(index_shift)
            R_squared_matrix(rows, cols) = 0; 
            index_matrix(rows, cols) = 0;
        else
            %assign the maximum value of the neuron to the the R_squared
            %matrix and tot the index_matrix
            R_squared_matrix(rows, cols) = maximumvalue_neuron;
            index_matrix(rows, cols) = index_shift;
        end
    end
end
%%
figure(5)
clear_R_squared = R_squared_matrix;
clear_R_squared (clear_R_squared >.3) = 0.3;
%subplot(6, 5, i);
imagesc(clear_R_squared )
clear_R_squared (isnan(clear_R_squared ))=0;
imagesc(clear_R_squared )
colors = colormap;
%set nan value a different color
colors(1, :) =  [0.5 0.5 0.5]; 
colormap(colors);
colorbar;
xlabel('columns')
ylabel('rows')
%imagesc(clear_R_squared,'AlphaData',~isnan(clear_R_squared))

%end 
figure(6)
colors = colormap;
%set nan value a different color
colors(1, :) =  [0.5 0.5 0.5]; 
colormap(colors);
imagesc(index_matrix)
colorbar;
xlabel('columns')
ylabel('rows')

%%
R_all_columns= mean(R_all,1);
for i= 1:30
    %[maximum_neuron_overtime, index ] = max(r2_mean_all(i,:));
    [maximum_neuron_overtime, index ] = max(R_all_columns(:,i,:), [], 2);


%preprocess R_squared for the location matrixes over different time points
maximum_neuron_overtime= maximum_neuron_overtime(:, 1:end)';
%the n inconsistent
%maximum_neuron_overtime= maximum_neuron_overtime(maximum_neuron_overtime>.05)
%best_r= horzcat(index, maximum_neuron);



%concatenatet the best_r with S1_unit_guide
matrix_info_overtime= horzcat(maximum_neuron_overtime, S1_unit_guide);
%remove the third column
matrix_info_overtime= matrix_info_overtime(:, 1:2);
%initiliase an array with maximum value for the each neuron or the maximum
%shift
max_neuron_time = zeros(96, 1);


%use accumarray to group the first and second collumn and find its max
%between them
%scaling_factor = 1000;
%matrix_info_overtime(:,1)= matrix_info_overtime(:, 1)* 1000;
%matrix_info_overtime(matrix_info_overtime(:, 1) < 0.2, 1) = 1;
%scaled = round(matrix_info_overtime(:, 1) * scaling_factor);
grouped_vals = accumarray(matrix_info_overtime(:,2), matrix_info_overtime(:, 1), [], @max);
%grouped_vals = accumarray(matrix_info_overtime(:, 1), matrix_info_overtime(:, 2));
%grouped_vals= find(max(matrix_info_overtime(:, 1),matrix_info_overtime(:, 2)))
%set the maximum value of the grouped_vals to each row coressponding to the channel
%grouped_vals= grouped_vals/scaling_factor
max_neuron_time(1:size(grouped_vals, 1)) = grouped_vals; 







%%assign the channels to the values
all_rows = size(max_neuron_time, 1);
all_indices = (1:all_rows);
%concatenate the channel with the max_neuron
max_neuron_time= horzcat(all_indices', max_neuron_time);



%% display the maximum R_squared in the matrix 
%initialise R_squared_matrix
R_squared_overtime= zeros(size(matrix_location));

%loop over the rows of the matrix_location
for rows = 1:size(matrix_location, 1)
    %loop over the columns of the matrix_location
    for cols = 1:size(matrix_location, 2)
        %compare the channel of max_neuron with the matrix location and obtain 
        %the value of that channel out of the second column of max_neuron.
        maximumvalue_neuron_time = max_neuron_time(max_neuron_time(:, 1) == matrix_location(rows, cols), 2);
       %account for empty values 
        if isempty(maximumvalue_neuron_time)
            R_squared_matrix1(rows, cols) = 0; 
        else
            %assign the maximum value of the neuron to the the R_squared
            %matrix and tot the index_matrix
            R_squared_matrix1(rows, cols) = maximumvalue_neuron_time;

        end
    end
end
figure(7)
shiftclear_R_squared = R_squared_matrix1;
shiftclear_R_squared(shiftclear_R_squared >.3) = 0.3;
subplot(5, 6, i);
imagesc(shiftclear_R_squared )
shiftclear_R_squared (isnan(shiftclear_R_squared ))=0;
colors = colormap;
%set nan value a different color
colors(1, :) =  [0.5 0.5 0.5]; 
colormap(colors);
imagesc(shiftclear_R_squared )
title(i)

if i ==19
    figure(8)
    shiftclear_R_squared = R_squared_matrix1;
    shiftclear_R_squared(shiftclear_R_squared >.3) = 0.3;
    imagesc(shiftclear_R_squared)
    shiftclear_R_squared (isnan(shiftclear_R_squared ))=0;
    colors = colormap;
    %set nan value a different color
    colors(1, :) =  [0.5 0.5 0.5]; 
    colormap(colors);
    imagesc(shiftclear_R_squared)
    colorbar;
end
end 
%colorbar;


%%
%taking the mean of each column
index_matrix(index_matrix == 0) = NaN;
column_index= nanmean(index_matrix);
%taking the mean of each row
row_index= nanmean(index_matrix,2);

%taking the mean of each column of the R_squared to look at where it best
%performs
R_squared_matrix(R_squared_matrix == 0) = NaN;
column_R= nanmean(R_squared_matrix);
%taking the mean of each row
row_R= nanmean(R_squared_matrix,2);

%%

%visually indexing 
figure(9);
index_visual_matrix = zeros(10,10);
index_visual_matrix(isnan(index_matrix)) = 0;
subplot(1, 3, 1);
index_visual_matrix(index_matrix <= 14) = 1;
index_visual_matrix(index_matrix > 14 ) = 2;
colors = colormap;
%set nan value a different color
colors(1, :) =  [0.5 0.5 0.5]; 
colormap(colors);
imagesc(index_visual_matrix);
colorbar;
title('<=- 49 sensory feedback')
xlabel('columns')
ylabel('rows')
subplot(1, 3, 2);
index_visual_matrix(index_matrix <= 16) = 1;
index_visual_matrix(index_matrix > 16 ) = 2;
colors = colormap;
%set nan value a different color
colors(1, :) =  [0.5 0.5 0.5]; 
colormap(colors);
imagesc(index_visual_matrix);
colorbar;
title('<= -29.4 sensory feedback')
xlabel('columns')
ylabel('rows')
subplot(1, 3, 3);
index_visual_matrix(index_matrix <= 18) = 1;
index_visual_matrix(index_matrix > 18 ) = 2;
imagesc(index_visual_matrix);
colors = colormap;
%set nan value a different color
colors(1, :) =  [0.5 0.5 0.5]; 
colormap(colors);
colorbar;
title('<= -9.8 sensory feedback')
xlabel('columns')
ylabel('rows')
%legend(index_visual_matrix, {'Nan','sensory feedback','efference copy'})

%%
figure(10)
%concatenatet the best_r with S1_unit_guide
histogram_info= horzcat(best_r, S1_unit_guide);
%remove the fourth column
histogram_info= histogram_info(:, 1:2);

time_shift = unique(histogram_info(:, 1));  
maximum_r_at_time_shift = zeros(size(time_shift));  

for i = 1:length(time_shift)
    %take the mean for multiple values of the same shift
    maximum_r_at_time_shift(i) = mean(histogram_info(histogram_info(:, 1) == time_shift(i), 2)); 
end

bar(time_shift, maximum_r_at_time_shift);
xlabel("time shift")
ylabel('R2 values')
title("max R2 at the time shift")
%%
figure(11)
shifts_number_val = histogram_info(:, 1);  

hist_bar = histogram(shifts_number_val);
count = hist_bar.BinCounts;
edges = hist_bar.BinEdges;
shift = (edges(1:end-1) + edges(2:end)) / 2;
xlabel('shifts');
ylabel('count');
title('number of neurons in each bin')


%%plotting


%%only print first 5 just to check
%for l= 1:5
%figure(l);
%plot(test_f_rates(:,l), 'b-');
%hold on;
%plot(test_pred(:,l), 'r-');
%hold off;
%xlabel('time in seconds') 
%ylabel('firing rate') 
%title('actual firing rate and predicted firing rate') 
%legend('measured firing rate', 'predicted firing_rate')

%end


%% Function that implements the ridge regression

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