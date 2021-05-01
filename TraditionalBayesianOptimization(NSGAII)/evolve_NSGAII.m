function  [realIndividual] = evolve_NSGAII(dmodel,Archive_train,problem,N,m)
w = 0;
wmax = 20;
%% Generate random population
    [Population,Boundary,~] = P_objective('init',problem,m,N);
     D = size(Boundary,2);
     %模型估计
     fitness = zeros(N,m);
     ER = fitness; s = ER; EIP = fitness; EIO = EIP;
     f_min = min(Archive_train(:,D+1:D+m),[],1);
     for i = 1:N
         for t = 1:m
              [fitness(i,t),~,ER(i,t)] = predictor(Population(i,:),dmodel(t));
               s(i,t) = sqrt(ER(i,t));
                EIP(i,t) = (f_min(1,t) - fitness(i,t)) * Gaussian_CDF((f_min(1,t)-fitness(i,t))./ s(i,t))...
                    + s(i,t) * Gaussian_PDF((f_min(1,t)-fitness(i,t))./ s(i,t));
         end
     end
%     [Population,EIP,FrontNo,CrowdDis] = EnvironmentalSelection(Population,EIP,N);
     [~,~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,EIP,N); % NSGAII原来的代码

    %% Optimization
    while w <= wmax
        MatingPool = TournamentSelection(2,N,FrontNo,-CrowdDis);
        Offspring  = GA(Population(MatingPool',:),Boundary);

%         Population = [Population; Offspring];
        
        for i = 1 : size(Offspring,1)
            for t= 1:m
                [fitness(i,t),~,ER(i,t)] = predictor(Offspring(i,:),dmodel(t));
                s(i,t) = sqrt(ER(i,t));
                EIO(i,t) = (f_min(1,t) - fitness(i,t)) * Gaussian_CDF((f_min(1,t)-fitness(i,t))./ s(i,t))...
                    + s(i,t) * Gaussian_PDF((f_min(1,t)-fitness(i,t))./ s(i,t));
            end
        end
        EI = [EIP;EIO];
        Population = [Population; Offspring];
        [Population,EIP,FrontNo,CrowdDis] = EnvironmentalSelection(Population,EI,N);
        w = w + 1;
    end
    %%  first front
     realIndividual = Population(find(FrontNo==1),:);
      realIndividual = unique(realIndividual,'rows');
     %% m+1
%      Population = Population(find(FrontNo==1),:);
%      EI = EIP(find(FrontNo==1),:);
%      EInum = size(EI,1)
%      EI_mid = zeros(1,m);
%      if EInum > m+1
%          for k = 1:m
%              [~,ind] = min(EI(:,k));
%              %注意EI值会有负值，是因为模型预测值会有负值
%              EI_mid(1,k) = (1/2)*max(EI(:,k));
% %              EI_mid(1,k) = (1/2)*(max(EI(:,k))-min(EI(:,k)));
%              realIndividual(k,:) = Population(ind,:);
%          end
%          [~,index] = min(dist(EI,EI_mid'));
%          realIndividual(m+1,:) = Population(index,:);
%      else
%          realIndividual = Population;
%      end
%      realIndividual = unique(realIndividual,'rows');

end