% load images and convert to double
img_paths = {'/MATLAB Drive/Lab4Mini/ben1.pgm', ...
             '/MATLAB Drive/Lab4Mini/ben2.pgm', ...
             '/MATLAB Drive/Lab4Mini/malg1.pgm', ...
             '/MATLAB Drive/Lab4Mini/malg2.pgm'};

% set block size for dividing the image
block_size = 32;  % adjust as needed
lattice_sizes = [5, 10, 25];  % lattice sizes to test

compression_ratios = [];  % array to store compression ratios

for img_index = 1:length(img_paths)
    % load current image and convert to double
    current_img = double(imread(img_paths{img_index}));
    [rows, cols] = size(current_img);

    % initialize array for storing image blocks
    blocks = [];  
    for i = 1:block_size:rows-block_size+1
        for j = 1:block_size:cols-block_size+1
            block = current_img(i:i+block_size-1, j:j+block_size-1);
            block_vector = reshape(block, 1, []);  % flatten block
            blocks = [blocks; block_vector];
        end
    end

    % loop through each lattice size for current image
    for n = lattice_sizes
        % create and train the som
        net = selforgmap([n n]);
        net = train(net, blocks');  % transpose for input format

        % initialize output image for compressed version
        compressed_img = zeros(size(current_img));  
        k = 1;

        % loop through blocks to find closest neuron and replace
        for i = 1:block_size:rows-block_size+1
            for j = 1:block_size:cols-block_size+1
                block_vector = blocks(k, :);
                neuron_idx = vec2ind(net(block_vector'));  % find winning neuron
                compressed_block = reshape(net.IW{1}(neuron_idx, :), block_size, block_size);
                compressed_img(i:i+block_size-1, j:j+block_size-1) = compressed_block;
                k = k + 1;
            end
        end

        % display original and compressed images for each configuration
        figure;
        subplot(1, 2, 1);
        imshow(current_img, []);
        title(['Original Image (Image ' num2str(img_index) ')']);
        
        subplot(1, 2, 2);
        imshow(compressed_img, []);
        title(['Compressed (Image ' num2str(img_index) ') with ' num2str(n) 'x' num2str(n) ' Lattice']);
        
        % calculate and store compression ratio
        unique_vectors_original = unique(blocks, 'rows');
        unique_vectors_som = unique(reshape(compressed_img, [], 1));
        compression_ratio = length(unique_vectors_som) / length(unique_vectors_original);
        compression_ratios = [compression_ratios; img_index, n, compression_ratio];  % store ratio with identifiers
        
        fprintf('Compression ratio for Image %d with %dx%d lattice: %.2f\n', img_index, n, n, compression_ratio);
    end
end

% plot compression ratios for each image
figure;
for img_index = 1:length(img_paths)
    img_ratios = compression_ratios(compression_ratios(:,1) == img_index, 3);
    plot(lattice_sizes, img_ratios, '-o', 'DisplayName', ['Image ' num2str(img_index)]);
    hold on;
end
xlabel('Lattice Size (NxN)');
ylabel('Compression Ratio');
title('Compression Ratio vs. Lattice Size for Each Image');
legend show;
grid on;
