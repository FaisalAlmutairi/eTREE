% A demo for the eTREE model (Eq. 8) in the following paper:
% 
% Almutairi, F. M., Wang, Y., Wang, D., Zhao, E., and Sidiropoulos, N. D., 
% "eTREE: Learning Tree-structured Embeddings", AAAI 2021
% 
% 
% (c) Faisal M. Almutairi (almut012@umn.edu), Sep 2021
%
% 
% 
% NOTE: this demo only runs on one train-test split; however, in the paper 
%       we average over 5 folds, where we train on 4 folds and test on the 
%       5th held out fold.
% 
clear; clc

load(strcat('MovieLens.mat'))
x_min = 1; % minimum value in the data
x_max = 5; % maximum value in the data
[N, M] = size(X_full);

% define the range of values for all the hyper-parameters
R = [25, 50, 100]; % matrix factorization rank
lbd  = [1e-3, 0.5, 1, 5, 10, 15, 20]; %  low-rankness regularization
mu   = [1e-3, 0.5, 1, 5, 10, 15, 20];  % Tree regularization
eta = 1000; % the regularization defined in Eq, 8 in the paper
Q = [2, 3, 4]; % the number of tree layers
Mq = cell(1,length(Q)); % cell data that contains the number of nodes in each layer
Mq{1} = [M, 10]; % number of nodes in each layer when Q = 2
Mq{2} = [M, 25, 5]; % number of nodes in each layer when Q = 3
Mq{3} = [M, 50, 10, 3]; % number of nodes in each when for Q = 4

maxNumIter = 1000; 
admmNumIter = 5;
treeNumIter = 5;
%% run cross-validation to find the best value for each hyper-parameter

% some variables for the cross-validation loops
Nrank = length(R); 
Nlbd = length(lbd); 
Nmu = length(mu); 
Nlayer = length(Q); 

RMSE_val_NMF = Inf(Nrank,Nlbd); % to save the NMF rmse for the validation 
RMSE_val_eTREE = Inf(Nrank,Nlbd,Nmu,Nlayer); % to save the eTREE rmse for the cross validation 
% cross-validation loops
parfor i = 1:Nrank
 for j = 1:Nlbd     
     %% NMF for initialization
     disp(['**I = ', num2str(i), '**J = ', num2str(j)])           
     disp('Training: running NMF')
     tic
     % run NMF to initialize for A and B_1
     [A_NMF, B_NMF, ~, rmse_NMF] = NMF_LowRank(X_full, R(i), lbd(j), admmNumIter, ... 
                maxNumIter, mask_train, mask_valid);
     toc
     RMSE_val_NMF(i,j) = rmse_NMF;
     % if the NMF fails, we use random initialization for A and B_1
     if rmse_NMF==Inf         
         A_init =  rand(N,R(i));    A_init = A_init/diag(sqrt(sum(A_init.^2)));
         B_init =  rand(M(1),R(i)); B_init = B_init/diag(sqrt(sum(B_init.^2)));
     else
         A_init = A_NMF;
         B_init = B_NMF;
     end
     
     for k = 1:Nmu
         for p = 1:Nlayer              
             % run eTRee for each hyper parameter value
             disp(['**I = ', num2str(i), '**J = ', num2str(j), '**K = ', ...
                 num2str(k),  '**P = ', num2str(p)])           
             disp('Training: running eTREE')
             tic
             [~, ~, ~, ~, ~, rmse_eTREE] = eTREE(X_full, A_init, B_init, Q(p),Mq{p}, ...
                 mu(k), lbd(j), eta, admmNumIter, treeNumIter, maxNumIter, mask_train, mask_valid);
             toc
             RMSE_val_eTREE(i,j,k,p) = rmse_eTREE;
         end
     end
 end
end
%% test on eTREE after we obtain the best hyper-parameters using cross-validation
disp('Testing for eTRee')
[i, j, k, p] = ind2sub(size(RMSE_val_eTREE),find(RMSE_val_eTREE==min(RMSE_val_eTREE(:)),1,'first'));

% run NMF for to initializa A and B_1
[A_NMF, B_NMF] = NMF_LowRank(X_full, R(i), lbd(j), admmNumIter, ...
             maxNumIter, mask_train, mask_valid);
% run eTREE to get the best number of iterations         
[~, ~, ~, ~, ~, ~, best_iter_eTRee] = eTREE(X_full, A_NMF, B_NMF,...
    Q(p), Mq{p}, mu(k), lbd(j), eta, admmNumIter, treeNumIter, maxNumIter, mask_train, mask_valid);
% Get the final trained model with the best hyper-parameters 
[A, B, D, S, ] = eTREE(X_full, A_NMF, B_NMF, Q(p), Mq{p}, ...
    mu(k), lbd(j), eta, admmNumIter, treeNumIter, best_iter_eTRee, mask_train);
X_eTREE = A*B{1}'*D;
RMSE_test_eTree = RMSE(X_full(mask_test), X_eTREE(mask_test), x_min, x_max)