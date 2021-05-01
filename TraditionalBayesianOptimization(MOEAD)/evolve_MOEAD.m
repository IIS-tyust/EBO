function  [realIndividual] = evolve_MOEAD(dmodel,Archive_train,problem,N,m)
    w = 0;
    wmax = 20;

   %% Parameter setting
    type = 2;

    %% Generate the weight vectors
    [W,N] = UniformPoint(N,m);
    T = ceil(N/10);

    %% Detect the neighbours of each solution
    B = pdist2(W,W); %计算W之间的欧氏距离
    [~,B] = sort(B,2); %B中的各行元素按升序排序
    B = B(:,1:T); %取每行最小的T个距离值的W索引号
    
    %% Generate random population
    [Population,Boundary,~] = P_objective('init',problem,m,N);
     D = size(Boundary,2);
     %模型估计(EI目标空间)
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
     Z = min(EIP,[],1);
    %% Optimization
    Offspring = zeros(N,D);
    fitnessO = zeros(N,m);
    while w <= wmax
        % For each solution
        for i = 1 : N      
            % Choose the parents
            P = B(i,randperm(size(B,2))); %第i个领域中的参考向量号随机排序

            % Generate an offspring
            Offspring(i,:) = GAhalf(Population(P(1:2),:),Boundary); 
            
%             for i = 1: size(Offspring,1)
                for t= 1:m
                    [fitnessO(i,t),~,ER(i,t)] = predictor(Offspring(i,:),dmodel(t));
                    s(i,t) = sqrt(ER(i,t));
                    EIO(i,t) = (f_min(1,t) - fitnessO(i,t)) * Gaussian_CDF((f_min(1,t)-fitnessO(i,t))./ s(i,t))...
                        + s(i,t) * Gaussian_PDF((f_min(1,t)-fitnessO(i,t))./ s(i,t));
                end
%             end
            % Update the ideal point
            Z = min(Z,EIO(i,:));

            % Update the neighbours
            switch type
                case 1
                    % PBI approach
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((EIP(P,:)-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((EIO(i,:)-Z).^2,2));
                    CosineP = sum((EIP(P,:)-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(EIO(i,:)-Z,T,1).*W(P,:),2)./normW./normO;
                    g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                    g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                case 2
                    % Tchebycheff approach
                    g_old = max(abs(EIP(P,:)-repmat(Z,T,1)).*W(P,:),[],2);
                    g_new = max(repmat(abs(EIO(i,:)-Z),T,1).*W(P,:),[],2);
                case 3
                    % Tchebycheff approach with normalization
                    Zmax  = max(EIP,[],1);
                    g_old = max(abs(EIP(P,:)-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(P,:),[],2);
                    g_new = max(repmat(abs(EIO(i,:)-Z)./(Zmax-Z),T,1).*W(P,:),[],2);
                case 4
                    % Modified Tchebycheff approach
                    g_old = max(abs(EIP(P,:)-repmat(Z,T,1))./W(P,:),[],2);
                    g_new = max(repmat(abs(EIO(i,:)-Z),T,1)./W(P,:),[],2);
            end
            Population(P(g_old>=g_new),:) = repmat(Offspring(i,:),length(P(g_old>=g_new)),1);
            EIP(P(g_old>=g_new),:) = repmat(EIO(i,:),length(P(g_old>=g_new)),1);
        end
        w = w + 1;
    end

   %%  first front
    [FrontNo,MaxFNo] = NDSort(EIP,inf); %输出第一层的非支配解
    realIndividual = Population(find(FrontNo==1),:);
    realIndividual = unique(realIndividual,'rows');
     %% m+1
%      [FrontNo,MaxFNo] = NDSort(EIP,inf); %输出第一层的非支配解
%      PopulationFirst = Population(find(FrontNo==1),:);
%      EI = EIP(find(FrontNo==1),:);
%      EInum = size(EI,1)
%      EI_mid = zeros(1,m);
%      if EInum > m+1
%          for k = 1:m
%              [~,ind] = min(EI(:,k));
%              EI_mid(1,k) = (1/2)*max(EI(:,k));
% %              EI_mid(1,k) = (1/2)*(max(EI(:,k))-min(EI(:,k)));
%              realIndividual(k,:) = PopulationFirst(ind,:);
%          end
%          [~,index] = min(dist(EI,EI_mid'));
%          realIndividual(m+1,:) = PopulationFirst(index,:);
%      else
%          realIndividual = PopulationFirst;
%      end
%      realIndividual = unique(realIndividual,'rows');

end