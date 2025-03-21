function problem22_inpainting()
    % Load data
    load('data22.mat');
    load('data21.mat');
    
    %Various N valuse that we will use.
    %Idealy use one by oe
    N_values = [500];
    
    % Number of random initializations we will test
    num_initializations = 6;
    %run for each N. For best results pick a single N
    for n_idx = 1:length(N_values)
        
        % space for images
        reconstructed_images = zeros(28, 28, 4);

        %HYPERPARAMS
        N = N_values(n_idx);
        max_iter = 600;
        learning_rate = 0.001;
        
        %  loss history
        all_loss_history = zeros(max_iter, 4);
        
        % best Z 
        best_Z = zeros(10, 4);
        
        % space for partial  images
        partial_images = zeros(28, 28, 4);
        %Process column by column as indicated in the problem question
        for col = 1:4
            % Extract partial y from the non noisy part
            y = X_n(1:N, col);
            
            % Create partial image
            partial_image = zeros(28, 28);
            partial_image(1:N) = y;
            partial_images(:,:,col) = partial_image;
            
            % Variables to track best loss
            best_loss = inf;
            
            % Multiple random inits
            for init = 1:num_initializations
                % Initialize Z randomly
                Z = randn(10, 1);
                
                % Loss history for this column
                loss_history = zeros(1, max_iter);
                
                % Gradient Descent to find optimal Z
                for iter = 1:max_iter
                    % Forward pass
                    W1 = A_1 * Z + B_1;
                    Z1 = max(W1, 0);  % ReLU
                    W2 = A_2 * Z1 + B_2;
                    X = 1 ./ (1 + exp(W2));
                    
                    % Create transformation matrix T
                    %As we said in proble question TX+noise should approximate y
                    T = [eye(N), zeros(N, 784-N)];
                    
                    % Compute loss 
                    loss = N*log10(norm(T * X - y)^2)+norm(Z)^2;
                    loss_history(iter) = loss;
                    
                    % Compute gradient 
                    % Gradient of loss w.r.t. X
                    dLdX = (2 * N / log(10)) * (T' * (T * X - y)) / norm(T * X - y)^2;
                    
                    % Gradient of X w.r.t. W2 
                    dXdW2 = -exp(W2) ./ (1 + exp(W2)).^2;  % Sigmoid grad
                    
                    % Gradient of loss w.r.t. W2 
                    dLdW2 = dLdX .* dXdW2;
                    
                    % Backpropagation
                    dLdZ1 = A_2' * dLdW2;
                    dLdZ1 = dLdZ1 .* (Z1 > 0);  % ReLU grad
                    
                    % Gradient w.r.t. Z
                    dLdZ = A_1' * dLdZ1;
                    
                    % Update Z
                    Z = Z - learning_rate * dLdZ;
                end
                
                % get last loss
                final_loss = loss_history(end);
                
                % Update best Z if this initialization has lower last loss
                if final_loss < best_loss
                    best_loss = final_loss;
                    best_Z(:, col) = Z;
                    all_loss_history(:, col) = loss_history;
                end
            end
            
            % Use the best Z we found to get the final reconstructed image
            Z = best_Z(:, col);
            W1 = A_1 * Z + B_1;
            Z1 = max(W1, 0);
            W2 = A_2 * Z1 + B_2;
            X = 1 ./ (1 + exp(W2));
            reconstructed_images(:,:,col) = reshape(X, 28, 28);
        end
        
             % Plot loss for columns side by side with smoothing
        figure;
        for col = 1:4
            subplot(1, 4, col);
            
            % Original loss
            loss_data = all_loss_history(all_loss_history(:,col) > 0, col);
            
            % Smoothed loss using moving average
            window_size = 20;  % Adjust this for different smoothing levels
            smoothed_loss = movmean(loss_data, window_size);
            
            % Plot both original and smoothed loss
            hold on;
            plot(loss_data, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Original Loss');
            plot(smoothed_loss, 'r', 'LineWidth', 2, 'DisplayName', 'Smoothed Loss');
            hold off;
            
            title(sprintf('Column %d Loss', col));
            xlabel('Iteration');
            ylabel('Loss');
            legend('show', 'Location', 'best');
        end
        sgtitle(sprintf('Loss Convergence (N = %d)', N));
        
        % Visualize results for this N
        figure;
        for col = 1:4
            subplot(3, 4, col);
            imshow(partial_images(:,:,col), []);
            title(sprintf('Partial Image (N=%d) %d', N, col));
            
            subplot(3, 4, col+4);
            imshow(reconstructed_images(:,:,col), []);
            title(sprintf('Reconstructed (N=%d) %d', N, col));
            
            subplot(3, 4, col+8);
            imshow(reshape(X_i(:,col), 28, 28), []);
            title(sprintf('Original %d', col));
        end
        sgtitle(sprintf('Inpainting Results for N = %d', N));
    end
end