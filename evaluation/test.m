% Open the file for writing
fileID = fopen('/rds/general/user/jl2622/home/HRTF_GAN/mat.txt', 'w');

% Write the text to the file
fprintf(fileID, 'Hello World');

% Close the file
fclose(fileID);