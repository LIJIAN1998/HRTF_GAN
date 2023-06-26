% function [num, s] = myfunc(n, m)
%     num = 10 * n
%     s = "kunkun " + m
% end

fileID = fopen('mat.txt', 'w');
fprintf(fileID, 'hello world'); 
fclose(fileID); 
