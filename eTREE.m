function [A, B, D, S, costValue, rmseValid, bestNumIter] = eTREE(Xdata, initA, initB1, ...
            Q, Mq, mu, lambda, eta, admmNumIter, treeNumIter, maxNumIter, maskTrain, maskValid)
%% 
% This code solves the eTREE model (Eq. 8) in the following paper:
% 
% Almutairi, F. M., Wang, Y., Wang, D., Zhao, E., and Sidiropoulos, N. D., 
% "eTREE: Learning Tree-structured Embeddings", AAAI 2021
% 
% The code follows the same notations as in the paper.
%
%
% The inputs of this function are as follows:
%       Xdata: the patient x item (or user x item) rating matrix of size N x M, with missing 
%               entries defined as NaN
%       initA: the initial value of the variable A
%       initB_1: the initia value of the variable B_1
%       Q: the number of tree layers
%       Mq: a vector that contains the number of nodes in each layer. 
%           Mq(1) is the number of nodes in layer 1 and so on. Note that
%           layer 1 is the very bottom layer.
%       mu, lambda, eta: are regularization hyper-parameters (as defined in Eq. 8)
%       admmNumIter: is the maximum number of iterations for the ADMM inner
%                 loops to solve the variables A and B1
%       treeNumIter: the maximum number of iterations for the tree inner
%                    loop that solves B_q (for q <= 2) and S_q (for all q)  
%       maxNumIter: is the maximum number of iteration of the main algorithm
%       maskTrain: is an NxM LOGICAL matrix with ones at the TRAIN data indices,
%                  and zeros otherwise.
%       maskValid: is an NxM LOGICAL matrix with ones at the VALIDATION data indices,
%                  and zeros otherwise.
% 
% 
% 
%  The outputs of this function are as follows:
%       A, B, D, S: are the variables as defined in Eq.8 i nthe paper
%       costValue: is the value of the cost function at every iteration   
%       rmseValid: is the Root Mean Square Error of the VLAIDATION data 
%       bestNumIter: is the iteration number before rmseValid starts
%                    increasing (the model overfits) 
% 
% 
% 
% 
% (c) Faisal M. Almutairi (almut012@umn.edu), Sep 2021
%% ================ define and initialize variables =======================
% define the training data
[N, ~] = size(Xdata);
X_train = NaN(N, Mq(1));
X_train(maskTrain) = Xdata(maskTrain);
X0 = X_train;
X0(~maskTrain) = 0;

% initialize variables A, Bq, D, Sq (and some ADMM intermediate variables)
B = cell(Q,1); % cell data that contains B_q for all q
S = cell(Q-1,1); % cell data that contains Z_q for all q
Z = cell(Q-1,1); % cell data that contains Z_q for all q

A =  initA;
B{1} =  initB1;
[~, R] = size(A);

Z{1} = zeros(Mq(1),R);
D = eye(Mq(1),Mq(1));
for j = 1:Mq(1)
    Z{1}(j,:) = B{1}(j,:)/norm(B{1}(j,:));
end
    
if Q > 1
    for q = 2:Q
        S{q-1} = zeros(Mq(q-1),Mq(q));
        for j = 1:Mq(q-1)            
            S{q-1}(j,randi(Mq(q))) = 1;
        end
        B{q} = rand(Mq(q),R); B{q} = B{q}/diag(sqrt(sum(B{q}.^2 ))); % these don't matter
        if (q<Q)
            for j = 1:Mq(q)
                Z{q}(j,:) = B{q}(j,:)/norm(B{q}(j,:)); % these don't matter
            end
        end
    end
end
A_old = A;
B_old = B{1};
U = zeros(N, R);
V = zeros(Mq(1),R);

% define some useful variables
admm_tol = 1e-3; 
cost_tol = 1e-07;
costValue = zeros(1, maxNumIter);
cost_old = 0;
diff = Inf;
RMSE_old = Inf;
outeriter = 0;

%% ======================= main algorithm ================================= 
while (outeriter < maxNumIter) && (diff > cost_tol)
    outeriter = outeriter + 1;
%     disp(outeriter)

    %% update A: ADMM
    DB = D*B{1}; 
    rho = trace(DB'*DB)/R;
    rho = rho/N;
    parfor i = 1:N
        DBi = DB(maskTrain(i,:),:);
        Li = chol(DBi'*DBi + (rho + lambda)*eye(R), 'lower');
        Fi = DB(maskTrain(i,:),:)'*X0(i,maskTrain(i,:))';
        for iter = 1:admmNumIter
            Yi = Fi + rho*A(i,:)' + rho*U(i,:)' ;
            A_tilde = Li'\(Li\Yi);         
            A(i,:) = max(0, A_tilde' - U(i,:));                                  
            U(i,:) = U(i,:) + A(i,:) - A_tilde';
            
            pAi = norm(A(i,:) - A_tilde','fro')^2/norm(A(i,:),'fro')^2;
            dAi = norm(A(i,:) - A_old(i,:),'fro')^2/norm(U(i,:),'fro')^2;            
            A_old(i,:) = A(i,:);
            if pAi < admm_tol && dAi < admm_tol
                break;
            end 
        end
    end
    %% update B_1 (B{1}): ADMM
    rho = trace(A'*A)/R;
    rho = rho/Mq(1);
    DZ = D*Z{1};
    DT = D*S{1}*B{2};
    B1 = B{1};
    parfor j = 1:Mq(1)
        Fj = A(maskTrain(:,j),:)'*X0(maskTrain(:,j),j); 
        if (Q==1)
            Lj = chol(A(maskTrain(:,j),:)'*A(maskTrain(:,j),:) + eye(R)*(eta + rho), 'lower');
        else
            
            Lj = chol(A(maskTrain(:,j),:)'*A(maskTrain(:,j),:) + eye(R)*(mu + eta + rho), 'lower');
        end
        for iter = 1:admmNumIter
            if (Q==1)
                Yj = Fj + eta*DZ(j,:)' + rho*B1(j,:)'*D(j,j) + rho*V(j,:)';
            else
                Yj = Fj + mu*DT(j,:)' + eta*DZ(j,:)' + rho*B1(j,:)'*D(j,j) + rho*V(j,:)';
            end
            B_tilde = Lj'\(Lj\Yj);
            B1(j,:) = max(0, (D(j,j)^-1)*(B_tilde' - V(j,:)));
            V(j,:) = V(j,:) + D(j,j)*B1(j,:) - B_tilde';

            pBj = norm(D(j,j)*B1(j,:) - B_tilde','fro')^2/norm(D(j,j)*B1(j,:),'fro')^2;       
            dBj = norm(B1(j,:) - B_old(j,:),'fro')^2/norm(V(j,:),'fro')^2;
            B_old(j,:) = B1(j,:);
            if pBj < admm_tol && dBj < admm_tol
                break;
            end
        end
    end
    B{1} = B1;
    %% update Z_1 (Z{1}) and D: Closed-forms
    for j = 1:Mq(1)
        Z{1}(j,:) = B{1}(j,:)/norm(B{1}(j,:));
        bj = B{1}(j,:)*A(maskTrain(:,j),:)';    
        bb = bj*bj';
        if bb == 0
            disp('deviding by zeros')
        end
        D(j,j) = (bj*X0(maskTrain(:,j),j))/(bb + eps);
        D(j,j) = max(D(j,j), 1e-30);
    end    
    %% Alternate between updating Sq and Bq         
    for iter = 1:treeNumIter
        for q = 2:Q
            if (q<Q)
                %% update Bq and Z_q (for q >= 2)
                Lq = chol(mu*S{q-1}'*S{q-1} + eye(Mq(q))*(mu + eta), 'lower');
                Yq = mu*S{q-1}'*B{q-1} + mu*S{q}*B{q+1} + eta*Z{q};
                B{q} = Lq'\(Lq\Yq);
                for j = 1:Mq(q)    
                    Z{q}(j,:) = B{q}(j,:)/norm(B{q}(j,:));
                end
            elseif (q==Q)
                for k = 1:Mq(Q)
                    Ik = any(S{Q-1}(:,k),2);
                    B{Q}(k,:) = sum(B{Q-1}(Ik,:),1)/sum(Ik);
                end
            end
            %% update S_q
            S{q-1} = zeros(Mq(q-1),Mq(q));            
            for j = 1:Mq(q-1)
                Kval = zeros(1,Mq(q));
                for k = 1:Mq(q)
                    Kval(k) = norm(B{q-1}(j,:) - B{q}(k,:));
                end
                idx = find(Kval == min(Kval), 1, 'first');
                S{q-1}(j,idx) = 1;
            end
        end
    end
    %% cost value
    TreeCost = 0;
    ZB_dif = 0;
    if Q > 1
        for q = 2:Q
            TreeCost = TreeCost + mu/2*norm(B{q-1} - S{q-1}*B{q},'fro')^2;
            ZB_dif = ZB_dif + norm(B{q-1} - Z{q-1},'fro')^2;
        end
    else 
        ZB_dif = norm(B{1} - Z{1},'fro')^2;
    end
    X_hat = A*B{1}'*D;
    costValue(outeriter) = 0.5*norm(maskTrain.*(X0 - X_hat), 'fro')^2 + TreeCost ...
                            + eta/2*ZB_dif + lambda/2*norm(A, 'fro'); 
    
    diff = abs(costValue(outeriter) - cost_old)/costValue(outeriter);
    cost_old = costValue(outeriter);
    if isnan(diff)
        rmseValid = Inf;
        disp('JONTEL fails for these params')
    end
    %% calculate RMSE on validation
    if (nargin > 12) && (mod(outeriter,5) == 0)
        x_min = 0; %min(min(X_train(maskTrain)));
        x_max = Inf; %max(max(X_train(maskTrain)));
        rmseValid = RMSE(Xdata(maskValid), (X_hat(maskValid)), x_min, x_max);
        disp(['*****eTREE: Validation RMSE: ', num2str(rmseValid), '*****'])
        % 1e-4 is a safe margin
        if rmseValid > RMSE_old + 1e-4
            disp('Stopped cause RMSE degraded')
            rmseValid = RMSE_old;    
            bestNumIter = outeriter - 5;
            break
        end
        RMSE_old = rmseValid;               
    end                  
    bestNumIter = outeriter; 
end

end