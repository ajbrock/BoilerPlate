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
%% Rawr
if 0
ur = unique(rank);
remove = unique(rank(ve(:,end)==0));
for i=1:length(remove)
    index = find(ur==remove(i));
    ur(index) = [];
end
order = zeros(100,length(ur));
for i = 1:100
for j = 1:length(ur)
    em(i,j) = mean(ve(rank==ur(j),i));
    estd(i,j) = std(ve(rank==ur(j),i));
end
[~,order(i,:) ] = sort(em(i,:));
end
%%
close all
for i = 1:100
%     plot(order(end,:),em(i,:),'r*')
    plot(em(i,order(end,:)),'r*')
    axis([0,length(ur),28,80]);
%     plot(order(end,:),order(i,:),'r*')
% axis([0,length(ur),0,length(ur)]);
    title(sprintf('Epoch %i/100',i))
    xlabel('Final rank');ylabel('Current rank');
%     axis([0,length(ur),0,length(ur)]);
    pause(0.1)
end
%%

end
%%
close all;
plot(ve(1,:))
plot(rank,ve(:,end)','*');
%%
ur = unique(rank);
for i = 1:length(ur)
    em(i) = mean(ve(rank==ur(i),end));
    estd(i) = std(ve(rank==ur(i),end));
end
%%
ve(em<20) =[];
ur(em<20) = [];
estd(em<20) = [];
seed(em<20) = [];
em(em<20) = [];

ur(estd>5) = [];
em(estd>5) = [];
estd(estd>5) = [];

% order = 
%%
errorbar(1:length(em),em,estd,'r*')
%%
rank(ve(:,end)==0)=[];
seed(ve(:,end)==0) = [];
ve(ve(:,end)==0,:)=[];

P=polyfit(rank',ve(:,end),1);
%%
figure
% plot(rank,ve(:,end),'*');
plot(ur,em,'*');
hold on;
plot([0,max(rank)],[P(2),P(2)+P(1)*max(rank)],'-r')
load errs_params
[emr,i] = sort(mean(e(:,[1,3,4])'));
stdr = std(e(:,[1,3,4])');
%%

% One figure option
figure
errorbarxy(100*emr(ur+1),em,100*stdr(ur+1),estd,{'r*','k','k'}); xlabel('SMASH Validation Error (%)');ylabel('True Validation Error (%)'); grid on;
P = polyfit(emr(ur+1),em,1);
ax = gca;
x = ax.XLim;
y = P*[x; 1,1]; 
hold on;
% plot(x,y,'--r')

12
%% Check error of each individual network?
% close all
% r = rank(seed==1)
% plot(e(ur),
e = e*100;
close all
plot(e(ur+1,1),em,'*',e(ur+1,2),em,'*',e(ur+1,3),em,'*',e(ur+1,4),em,'*'); hold on;
%%
ax = gca;
ax.ColorOrderIndex = 1;
% indices = e(ur+1,1) < .7;
% g = e(ur+1,1);
% % P = polyfit(g(indices)',em(indices),1)
% ax = gca;
% x = ax.XLim;
% y = P*[x; 1,1];
[ee,ind] = sort(e(ur+1,1));
eem = em(ind);
plot(ee(ee<75),em(ee<75));
P = polyfit(ee,em(ind)')
P = polyfit(ee,em(ind)'),
P1 = polyfit(e(ur+1,1)',em,1);
P2 = polyfit(e(ur+1,2)',em,1);
P3 = polyfit(e(ur+1,3)',em,1);
P4 = polyfit(e(ur+1,4)',em,1);
x = ax.XLim;
y1 = P1*[x; 1,1];
y2 = P2*[x; 1,1];
y3 = P3*[x; 1,1];
y4 = P4*[x; 1,1];
plot(x,y1,'--',x,y2,'--',x,y3,'--',x,y4,'--');

% plot(e(ur+1,2),em,'*');
% plot(e(ur+1,3),em,'*');
% plot(e(ur+1,4),em,'*');
legend('.646','.707','.594','.612')

% legend('1','2','3','4');
% legend(mean(e))
% plot(ve(:,

% axis([0,500,26,35]);
%% USE THIS TO SEE ERROR OVER TIME
if 0
close all
for i = 1:100
%     plot(order(end,:),em(i,:),'r*')
    plot(ve(:,end),ve(:,i),'r*');
    axis([min(ve(:,end)),max(ve(:,end)),min(ve(:,end)),55])
%     plot(order(end,:),order(i,:),'r*')
% axis([0,length(ur),0,length(ur)]);
    title(sprintf('Epoch %i/100',i))
    xlabel('Final error');ylabel('Current error');
%     axis([0,length(ur),0,length(ur)]);
    pause(0.1)
end
end

%% Get valid stamp at final epoch and train_stamp at first epoch
% figure
