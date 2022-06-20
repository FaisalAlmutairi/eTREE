function [A, B, cost_val, RMSE_new, best_iter] = NMF_LowRank(X_full, R, lbd, ADMMiter, max_iter, mask_train, mask_valid)



[N, M] = size(X_full);
X_train = NaN(N, M);
X_train(mask_train) = X_full(mask_train);

X0 = X_train;
X0(~mask_train) = 0;

% initialization
A =  rand(N,R); A = A/diag(sqrt(sum(A.^2)));
B =  rand(M,R); B = B/diag(sqrt(sum(B.^2)));  
A_old = A;
B_old = B;
U = zeros(N, R);
V = zeros(M(1),R);

admm_tol = 1e-3;
cost_tol = 1e-07;
cost_val = zeros(1, max_iter);

cost_old = 0;
diff = Inf;
RMSE_old = Inf;

outeriter = 0;
while (outeriter<max_iter) && (diff>cost_tol)
    outeriter = outeriter + 1;
%     dispj(outeriter)
    %% update A: ADMM
    rho = trace(B'*B)/R;
    rho = rho/N;
    parfor i = 1:N
        Bi = B(mask_train(i,:),:);
        Fi = B(mask_train(i,:),:)'*X0(i,mask_train(i,:))';
        Li = chol(Bi'*Bi + eye(R)*(lbd + rho), 'lower');
        for iter = 1:ADMMiter
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
    %% update B{1}: ADMM
    rho = trace(A'*A)/R;
    rho = rho/M(1);
    parfor j = 1:M
        Aj = A(mask_train(:,j),:);
        Fj = A(mask_train(:,j),:)'*X0(mask_train(:,j),j); 
        Lj = chol(Aj'*Aj + eye(R)*(lbd + rho), 'lower');
        for iter = 1:ADMMiter
            Yj = Fj + rho*B(j,:)' + rho*V(j,:)';
            B_tilde = Lj'\(Lj\Yj);
            B(j,:) = max(0, B_tilde' - V(j,:));
            V(j,:) = V(j,:) + B(j,:) - B_tilde';

            pBj = norm(B(j,:) - B_tilde','fro')^2/norm(B(j,:),'fro')^2;       
            dBj = norm(B(j,:) - B_old(j,:),'fro')^2/norm(V(j,:),'fro')^2;
            B_old(j,:) = B(j,:);
            if pBj < admm_tol && dBj < admm_tol
                break;
            end
        end
    end
    %% cost value
    X_hat = A*B';
    cost_val(outeriter) = 0.5*norm(mask_train.*(X0 - X_hat), 'fro')^2 ...
                            + lbd/2*(norm(A, 'fro')^2 + norm(B, 'fro')^2); 
    diff = abs(cost_val(outeriter) - cost_old)/cost_val(outeriter);
    cost_old = cost_val(outeriter);
    if isnan(diff)
        RMSE_new = Inf;
        disp('NMF fails for these params')
    end
    %% calculate RMSE on validation
    if (nargin > 6) && (mod(outeriter,5) == 0)
        RMSE_new = RMSE(X_full(mask_valid), (X_hat(mask_valid)), 0, Inf);
        disp(['*****NMF: Validation RMSE: ', num2str(RMSE_new), '*****'])
        % 1e-3 is a safe margin
        if RMSE_new > RMSE_old + 1e-4
            disp('Stooped cause RMSE degraded')
            RMSE_new = RMSE_old; 
            best_iter = outeriter - 5;
            break
        end
        RMSE_old = RMSE_new;        
    end
    best_iter = outeriter;
end


end