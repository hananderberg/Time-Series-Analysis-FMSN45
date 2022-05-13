function [u] = generateMarkov(steps)
    
    u = zeros(steps,1);
    
    v1 = rand(1);
    r = 0;
    if v1 < 0.5
        r = 1;
    end 
    u(1) = r;
    
    % Now start the loop
    for i=2:steps
        v = rand(1);
        if v < 1/8
            u(i) = abs(u(i-1)-1);
        else
            u(i) = u(i-1);
        end
    end 
end
