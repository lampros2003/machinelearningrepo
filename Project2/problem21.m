function problem21_generate_eights()
    % Load the generative model data
    load('data21.mat');
    %we initialize a seed at a specific value so htat the code is
    %reproducable
    rng(42);
    
    % Allocate space for 100 images
    generated_images = zeros(28, 28, 100);
    
    % Generate 100 realisations with the model
    for i = 1:100
        % Random in using the seed we chose before)
        Z = randn(10, 1);
        
        % First layer 
        W1 = A_1 * Z + B_1;
        Z1 = max(W1, 0);  % ReLU
        
        % Second layer
        W2 = A_2 * Z1 + B_2;
        X = 1 ./ (1 + exp(W2));  % Sigmoid 
        
        %Generate the final 28X28 image
        generated_images(:,:,i) = reshape(X, 28, 28);
    end
    
    % Create a 10x10 montage 
    montage_image = zeros(280, 280);
    for row = 1:10
        %print the images onmto the plot
        for col = 1:10
            idx = (row-1)*10 + col;
            montage_image((row-1)*28+1:row*28, (col-1)*28+1:col*28) = generated_images(:,:,idx);
        end
    end
    
    % Display 
    figure;
    imshow(montage_image, []);
    title('100 Generated Handwritten 8s');
    imwrite(montage_image, 'generated_eights_montage.png');
end