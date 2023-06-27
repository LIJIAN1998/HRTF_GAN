function [result_int, result_str] = multiply_and_concat(num, str)
    result_int = num * 10;
    result_str = ['kunkun', str];
end

% Call the function with the provided arguments
multiply_and_concat(input_int, input_str);