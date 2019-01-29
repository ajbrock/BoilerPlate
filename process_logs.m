%%

% NOTE: all logs except #1 are 0-indexed! (okay I changed #1 to be #0)
clc
clear all
close all
fclose all;
% Get all files
names = dir();
names = {names.name};
names = names(3:end);
j = 0;
for i = 1:length(names)
    if length(names{i})>5 && strcmp(names{i}(end-5:end),'.jsonl')
        j=j+1;
        f{j} = names{i};
    end
end
%%


% q=sscanf(s,'%3s_k%iL%i_ice%i_')%11s_seed%i_epochs%iC%i_log.jsonl')
%%
te = zeros(length(f),100); % training error
ve = zeros(length(f),100);


for i = 1:length(f)
    fid = fopen(f{i},'r');
    s = f{i};
    n = sscanf(s,'SMASHv7B_Main_SMASHv7B_D12_K4_N8_Nmax16_Rank%i_seed%i_100epochs_log.jsonl');
    rank(i) = n(1);
    seed(i) = n(2);

    while ~feof(fid);
        s = fgets(fid);
        train = strfind(s,'train_loss');
        valid = strfind(s,'val_loss');
        if train
            this_epoch = strfind(s,'"epoch"');
            epoch = sscanf(s(this_epoch:end),'"epoch": %i')+1;
            
            stamp = strfind(s,'"_stamp"');
            tt(i,epoch) = sscanf(s(stamp:end),'"_stamp": %f');
            
            train_loc = strfind(s,'"train_loss"');
            tl(i,epoch) = sscanf(s(train_loc:end),'"train_loss": %f');
            
            te_loc = strfind(s,'"train_err"');
            te(i,epoch) = sscanf(s(te_loc:end),'"train_err": %f');
            
        elseif valid
            this_epoch = strfind(s,'"epoch"');
            epoch = sscanf(s(this_epoch:end),'"epoch": %i')+1;
            
            stamp = strfind(s,'"_stamp"');
            tv(i,epoch) = sscanf(s(stamp:end),'"_stamp": %f');
            
            vl_loc = strfind(s,'"val_loss"');
            vl(i,epoch) = sscanf(s(vl_loc:end),'"val_loss": %f');
            
            ve_loc = strfind(s,'"val_err"');
            ve(i,epoch) = sscanf(s(ve_loc:end),'"val_err": %f');
            
            
        end
        %
    end
    fclose(fid);

    
    %     err(i) = ve(i,epochs(i));
end