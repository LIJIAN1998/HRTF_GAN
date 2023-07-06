function [pol_acc1, pol_rms1, querr1] = calc_loc(fullSofa1FileName, fullSofa2FileName)
    curdir = cd; amt_start(); cd(curdir); % start AMT
    pol_acc1_avg = 0.0;
    pol_rms1_avg = 0.0;
    querr1_avg = 0.0;
    Sofa1 = SOFAload(fullSofa1FileName);
    Sofa2 = SOFAload(fullSofa2FileName);
    [h1,fs,az,el] = sofa2hrtf(Sofa1);
    [h2,fs2,az2,el2] = sofa2hrtf(Sofa2);
    fs = 48000;
    fs2 = 48000;
    num_exp = 1;

    % Run barumerli2023 for h1
    disp('Running barumerli2023 for first HRTF...'), tic

    dtf = getDTF(h1,fs);
    SOFA_obj1 = hrtf2sofa(dtf,fs,az,el);
    [~, target1] = barumerli2023_featureextraction(SOFA_obj1, 'dtf', 'targ_az', SOFA_obj1.SourcePosition(:, 1), 'targ_el', SOFA_obj1.SourcePosition(:, 2));\
    
    dtf = getDTF(h2,fs2);
    SOFA_obj2 = hrtf2sofa(dtf,fs2,az2,el2);
    [template2, target2] = barumerli2023_featureextraction(SOFA_obj2, 'dtf', 'targ_az', SOFA_obj2.SourcePosition(:, 1), 'targ_el', SOFA_obj2.SourcePosition(:, 2));

    pol_acc1 = 0;
    pol_rms1 = 0;
    querr1 = 0;

