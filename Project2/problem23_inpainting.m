function problem23_inpainting()
    % Load data
    load('data23.mat');
    load('data21.mat');
    
    % We set N to 49 only as dictated in the problem
    N_values = [49]; 
    
    % Number of random initializations we will test
    num_initializations = 9;
    %run for each N. For best results pick a single N
    for n_idx = 1:length(N_values)
        N = N_values(n_idx);
        
        % space for images
        reconstructed_images = zeros(28, 28, 4);
        % Hyperparams  ,changed the learning rate,higher rate had better
        % results
            max_iter = 800;
            learning_rate = 0.001;
        % best Z 
        best_Z = zeros(10, 4);
        
        % space for images(low res) its the same dim but we paint 4X4
        % pixels
        low_res_inputs = zeros(28, 28, 4);
        
        % Store loss histories for each column
        loss_histories = zeros(max_iter, 4);
        
        for col = 1:4
            % Extract partial observation
            y = X_n(1:N, col);
            
            % Create transformation matrix T for low-res (same as before)
            T = zeros(N, 784);
            for i = 1:N
                %%find the rows ajd columns that are part of the grid
                grid_row = ceil(i / 7);
                grid_col = mod(i-1, 7) + 1;
                
                block_row_start = (grid_row-1)*4 + 1;
                block_col_start = (grid_col-1)*4 + 1;
                
                %each block has 16 indexes 
                block_indices = zeros(1, 16);

                %every index is comb of r,c
                %go through all combos
                for r = 0:3
                    for c = 0:3
                        %get each pixel
                        pixel_row = block_row_start + r;
                        pixel_col = block_col_start + c;
                        %use the relationship we were given in hints
                        pixel_index = (pixel_row-1)*28 + pixel_col;
                        block_indices((r*4)+c+1) = pixel_index;
                    end
                end
                %set that we must set to 1/16 to calc proper mean
                T(i, block_indices) = 1/16;
            end
            
            % Prepare low-resolution input 
            low_res_image = reshape(X_n(:,col), 7, 7);
            low_res_upscaled = kron(low_res_image, ones(4,4));
            low_res_inputs(:,:,col) = low_res_upscaled;
            
            % Variables to track best loss
            best_loss = inf;
            
            % Multiple random inits
            for init = 1:num_initializations
                % Inite Z
                Z = randn(10, 1);
                
                % Loss history for this initialization
                loss_history = zeros(1, max_iter);
                
                % Gradient Descent 
                for iter = 1:max_iter
                    % Forward pass
                    W1 = A_1 * Z + B_1;
                    Z1 = max(W1, 0);  % ReLU
                    W2 = A_2 * Z1 + B_2;
                    X = 1 ./ (1 + exp(W2));
                    
                    % Compute loss (mean squared error)
                    loss =  N*log10(norm(T * X - y)^2)+norm(Z)^2;
                    loss_history(iter) = loss;
                    
                    % Compute gradient
                    dLdX = (2 * N / log(10)) * (T' * (T * X - y)) / norm(T * X - y)^2;
                    dXdW2 = -exp(W2) ./ (1 + exp(W2)).^2;  % Sigmoid gradient
                    dLdW2 = dLdX .* dXdW2;
                    dLdZ1 = A_2' * dLdW2;
                    dLdZ1 = dLdZ1 .* (Z1 > 0);  % ReLU gradient
                    dLdZ = A_1' * dLdZ1;
                    
                    % Update Z
                    Z = Z - learning_rate * dLdZ;
                end
                
                % Find the final loss
                final_loss = loss_history(end);
                
                % Update best Z if this initialization has lower loss
                if final_loss < best_loss
                    best_loss = final_loss;
                    best_Z(:, col) = Z;
                    loss_histories(:, col) = loss_history';
                end
            end
            
            % Use the best Z to get the final reconstructed image
            Z = best_Z(:, col);
            W1 = A_1 * Z + B_1;
            Z1 = max(W1, 0);
            W2 = A_2 * Z1 + B_2;
            X = 1 ./ (1 + exp(W2));
            reconstructed_full = reshape(X, 28, 28);
            reconstructed_images(:,:,col) = reconstructed_full;
        end
        
        % Plot loss convergence side by side with smoothed loss
        figure;
        for col = 1:4
            subplot(2, 2, col);
            
            % Original loss plot
            plot(loss_histories(:, col), 'b-', 'LineWidth', 1);
            hold on;
            
            % Smoothed loss using running median
            window_size = 30; % Adjust this for different smoothing levels
            smoothed_loss = movmedian(loss_histories(:, col), window_size);
            plot(smoothed_loss, 'r-', 'LineWidth', 1.5);
            
            title(sprintf('Loss Convergence Column %d', col));
            xlabel('Iteration');
            ylabel('Loss');
            legend('Original Loss', 'Smoothed Loss');
            hold off;
        end
        sgtitle('Loss Convergence for All Columns');
        
        % Visualize results
        figure;
        for col = 1:4
            subplot(3, 4, col);
            imshow(low_res_inputs(:,:,col), []);
            title(sprintf('Low-Res Input %d', col));
            
            subplot(3, 4, col+4);
            imshow(reconstructed_images(:,:,col), []);
            title(sprintf('Reconstructed %d', col));
            
            subplot(3, 4, col+8);
            imshow(reshape(X_i(:,col), 28, 28), []);
            title(sprintf('Original %d', col));
        end
        sgtitle('Low-Resolution Image Reconstruction Results');
    end
end