function  [Population,prefitnessP,realIndividual] = KDPEA2(dmodel,Population,prefitnessP,Boundary,reference_vector,Archive_train)
% ������һ����������ʵ���۵ĸ���
%%
%-------------------��������----------------
format compact;
Coding = 'Real';
[N,m] = size(reference_vector);
D = size(Population,2);
ER = [];

%------------------��ÿ���ο������϶�Ӧ�Ĳο���(�����)-------
ref_point = reference_point_assign(prefitnessP,reference_vector);
%------------------SBX and PM ������ѡ��---------------
[MatingPool] = F_mating(Population);
%     MatingPool= F_mating_new(Population,N);
Offspring = P_generator(MatingPool,Boundary,Coding,N);
prefitnessO = zeros(size(Offspring,1),m);
ER = prefitnessO; %ER�������
%---------ʹ��Krilingģ�͹����Ӵ�(��ѡ��)-----------
for i = 1:size(Offspring,1)
    for k = 1:m
        [prefitnessO(i,k),~,~] = predictor(Offspring(i,:),dmodel(k));
    end
end
Of = [Offspring prefitnessO];
Xf = [Population prefitnessP];
f = [prefitnessP;prefitnessO];
XO = [Xf;Of];
%----------------���Ƕȹ����ο�����------------
indiv_ind = Angle_assign_refvec(f,reference_vector);%ÿ���ο����������ж���������
%------------------�����²ο���-----------------
ref_point_new = reference_point_assign(prefitnessO,reference_vector);
%--------------------����ÿ���ο������ϵĲο���------
ref_point = selection_reference_point(ref_point,ref_point_new);
%         % -------------------����ÿ���ο������ϵ������--------------
%         ideal_point = ideal_point_generation(indiv_ind,f,reference_vector);
%--------------------����ָ��ѡ�����-------------
%     Xf = selection_individual(indiv_ind,XO,reference_vector,ideal_point);
Xf = [];
Xf = selection_individual(indiv_ind,XO,reference_vector,ref_point);
Population = [];
prefitnessP = [];
Population = Xf(:,1:D); %����ָ��ѡ������ĸ�����Ϊ��һ������
prefitnessP = Xf(:,D+1:D+m);
%--------�齨EI���󣬸���EIֵ����ѡ��ϴ��1/2���������ʵ����
f_min = min(Archive_train(:,D+1:D+m),[],1);
% EI_times = [];
% prefitness = [];
for i = 1:size(Population,1)
    for t = 1:m
        [prefitness(i,t),~,ER(i,t)] = predictor(Population(i,:),dmodel(t));
        s(i,t) = sqrt(ER(i,t));
        EI(i,t) = (f_min(1,t) - prefitness(i,t)) * Gaussian_CDF((f_min(1,t)-prefitness(i,t))./ s(i,t))...
            + s(i,t) * Gaussian_PDF((f_min(1,t)-prefitness(i,t))./ s(i,t));
    end
    EI_min(i) = min(EI(i,:));
end
[EI_value,EI_id] = sort(EI_min,'descend');
% num = round((1/2)* size(EI_id,2));
num = 5;
realIndividual = Population(EI_id(1,1:num),:);
end





