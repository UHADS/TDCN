clear;close all;

folder = 'F:DIV2K\DIV2K_train_HR';

savepath = 'train0.h5';
size_input = 96;
size_label = 96;
stride = 57;
batchsize=32;
max_numPatches= 3000*batchsize;
aaa = max_numPatches/800;
%% scale factors
%scale = [2,3,4];
scale = 32;
%% downsizing
%downsizes = [1,0.7,0.5];
downsizes = 1;
%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
data1 = zeros(size_input, size_input, 1, 1);
label1 = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];
%filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];

for i = 1 : length(filepaths)
    count = 0;
    for flip = 1: 3 
        for degree = 1 : 4 
            for s = 1 : length(scale)
                for downsize = 1 : length(downsizes)
                    image = imread(fullfile(folder,filepaths(i).name));
                    
                    if flip == 1 % 
                        image = flipdim(image ,1);
                    end
                    if flip == 2 % 
                        image = flipdim(image ,2);
                    end
                    
                    image = imrotate(image, 90 * (degree - 1));%  

                 %   image = imresize(image,downsizes(downsize),'bicubic');
                    
                    if size(image,3)==3            
                        image = rgb2ycbcr(image);
                        image = im2double(image(:, :, 1));

                        im_label = modcrop(image, scale(s));
                        [hei,wid] = size(im_label);
                       im_input = im_label;
                        % im_input = imresize(imresize(im_label,1/scale(s),'bicubic'),[hei,wid],'bicubic');
                        filepaths(i).name
                        for x = 1 : stride : hei-size_input+1
                            for y = 1 :stride : wid-size_input+1

                                subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
                                subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                                
                                count=count+1;

                                data1(:, :, 1, count) = subim_input;
                                label1(:, :, 1, count) = subim_label;
                            end
                        end
                    end
                end    
            end
        end
    end
    bbb1 = (i-1)*aaa+1;
    bbb2 = i*aaa;
    order = randperm(count);
    data1 = data1(:, :, 1, order);
    label1 = label1(:, :, 1, order); 
    data(:,:,1,bbb1:bbb2) = data1(:,:,1,1:120);
    label(:,:,1,bbb1:bbb2) = label1(:,:,1,1:120);
end


%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;
subNum = max_numPatches;
for batchno = 1:floor(subNum/chunksz)
    batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
