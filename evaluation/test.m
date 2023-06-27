function [result_int, result_str] = test(num, str)
    result_int = num * 10;
    result_str = ['kunkun', str];

    disp(result_int);
    disp(result_str);
end
