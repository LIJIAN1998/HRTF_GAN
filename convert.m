% 导入 SOFA API
addpath('/rds/general/user/jl2622/home/SOFA_API')  % 替换为 SOFA API 的路径

% 读取 SOFA 文件
sofaFile = '/rds/general/user/jl2622/home/HRTF-projection/runs-hpc/ari-upscale-4/valid/nodes_replaced/sofa_min_phase/SONICOM_100.sofa';  % 替换为您的 SOFA 文件路径
sofaData = SOFAload(sofaFile);

% 提取音频数据
audioData = sofaData.Data.IR;  % 根据您的 SOFA 文件结构选择正确的音频数据

% 获取音频数据的采样率
sampleRate = sofaData.Data.SamplingRate;

% 将音频数据写入 WAV 文件
outputFile = '/rds/general/user/jl2622/home';  % 替换为输出 WAV 文件的路径
audiowrite(outputFile, audioData, sampleRate);

% 提示操作完成
disp('SOFA 文件已成功转换为 WAV 音频文件.');
