
%MATLAB code shared here is associated with the manuscript entitled
% "Hybridizing Levy Flight and Whale Optimization Algorithm for Effective Data Clustering”, 
% published in the Arabian Journal for Science and Engineering.
% The initial implementation is based on the Whale Optimization Algorithm (see Reference 17 in the manuscript). 
% The code was further modified by Dr. Ashish Kumar Sahu and Dr. Tribhuvan Singh to address data clustering problems. 
% To run the code, please ensure that the latest version of MATLAB is installed on your system.
tic; 
clc;
  clear;
  close all;
% initialize the location and Energy of the rabbit
st = tic;   % add in line number 6
 run=25;
 Best_Fitness=inf;
 output=[];
 Mean=[];
 WORST=[];
 BEST=[];
 dim=2;
 for it=1:run
  Leader_score=inf;
  data = load('Flame.mat');
  X = data.Flame;
  Y=X;  % for using in ARI, NMI and Purity.
  k=2;
   MaxIt=500; % Maximum Number of Iterations
 nPop=60; % Population Size (Star Size)
 Max_iter=MaxIt;
 N=nPop; 
 SearchAgents_no=N;
  X=X(:,1:2);
 CostFunction=@(m) ClusteringCost(m, X); % Cost Function
 VarSize=[k size(X,2)];
  nVar=prod(VarSize); % Number of Decision Variables

 VarMin= repmat(min(X),k,1); % Lower Bound of Variables
 VarMax= repmat(max(X),k,1); % Upper Bound of Variables
 lb=VarMin;
 ub=VarMax;
 empty_particle.Position=[];
 empty_particle.Cost=[];
 %empty_particle.Out=[];
 particle=repmat(empty_particle,nPop,1);
 particle1=repmat(empty_particle,nPop,1);
  for i=1:nPop

 % Initialize Position of Stars
 particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
 a=particle(i).Position;
 % pause;
 % Evaluation
 [particle(i).Cost]=CostFunction(particle(i).Position);

 
 % if condicion ok , the star is considered as the first ...
%Black Hole
 if particle(i).Cost < Leader_score
    Leader_score=particle(i).Cost;
    BestSol=particle(i);
    Leader_pos=particle(i).Position;
 end
  end
x1=  [particle.Position];
 x2=[particle.Cost];

 
%function [Leader_score,Leader_pos,Convergence_curve]=WOA(SearchAgents_no,Max_iter,lb,ub,dim,fobj)

% initialize position vector and score for the leader
%Leader_pos=zeros(1,dim);
 %change this to -inf for maximization problems


%Initialize the positions of search agents
%Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iter);

t=0;% Loop counter

% Main loop
while t<Max_iter
    
    
    a=2-t*((2)/Max_iter); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/Max_iter);
    
    % Update the Position of search agents 
    for i=1:N
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        
        
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        
        p = rand();        % p in Eq. (2.6)
        
      %  for j=1:dim
            
            if p<0.5   
                if abs(A)>=1
                    rand_leader_index = floor(SearchAgents_no*rand()+1);
                    X_rand=particle(rand_leader_index).Position;
                    D_X_rand=abs(C*particle(rand_leader_index).Position-particle(i).Position);% Eq. (2.7)
                  %  pos(:,j)=D_X_rand;
                    particle1(i).Position=X_rand-A*D_X_rand;      % Eq. (2.8)
                    
                elseif abs(A)<1
                    D_Leader=abs(C*Leader_pos-particle(i).Position); % Eq. (2.1)
                    particle1(i).Position=Leader_pos-A*D_Leader;      % Eq. (2.2)
                end
                
            elseif p>=0.5
              
                distance2Leader=abs(Leader_pos-particle(i).Position);
           
                 particle1(i).Position=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos+rand(1,dim).*Levy(dim);
                 
                
            end
            
      %  end
    end
    
    
     for i=1:N
        % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=particle1(i).Position>ub;
        Flag4lb=particle1(i).Position<lb;
        particle1(i).Position=(particle1(i).Position.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate objective function for each search agent
        particle1(i).Cost=CostFunction(particle1(i).Position);
     end
     
        merge=[particle;particle1];
        T = struct2table(merge); % convert the struct array to a table
        sortedT = sortrows(T, 'Cost'); % sort the table by 'Cost'
        MERGE = table2struct(sortedT); % change it back to struct array if necessary    
        % Update the leader
        
         particle=MERGE(1:N);  
         Leader_pos=particle(1).Position;
         Leader_score=particle(1).Cost;
if Best_Fitness > Leader_score
    Best_Fitness = Leader_score;
    Centroids=Leader_pos;
end
        
    
    
    
    t=t+1;
    Convergence_curve(t)=Leader_score;
    CNVG(t)=Leader_score;
    display(['At iteration ', num2str(t),  ' and run ', num2str(it), ' the best fitness is ', num2str(Leader_score)]);
   
end


y1=CNVG';
 worst=max(y1);
 WORST=[WORST,worst];
 best=min(y1);
 boxplot=[BEST,best];
 BEST=[BEST,best];
 m1=mean(y1);
 Mean=[Mean,m1];
 output=[output,y1];
 end
boxplot=boxplot';
STD=std(Mean);
MEAN=mean(Mean);
WORST=max(WORST);
BEST=min(BEST);
elapsedTime = toc(st);
FINAL=[BEST;WORST;MEAN;STD;elapsedTime];
 figure;
 plot(CNVG,'LineWidth',2);
 xlabel('Iteration');
 ylabel('Best Cost');
 [~, out] = ClusteringCost(Centroids, X);
 predicted_labels = out.ind;  % This is what you need
 true_labels = Y(:, end);
 % Now compute ARI, NMI and Purity
[NMI, ARI, Purity] = evaluate_clustering(true_labels, predicted_labels);
FINAL1=[ARI; NMI; Purity];
toc;
function [z, out] = ClusteringCost(m, X)

 % Calculate Distance Matrix
 d = pdist2(X, m);
 % Assign Clusters and Find Closest Distances
 [dmin, ind] = min(d, [], 2);
 %pause;
 % Sum of Within-Cluster Distance
 WCD = sum(dmin);
z=WCD;
out.d=d;
out.dmin=dmin;
out.ind=ind;
out.WCD=WCD;


end

function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
%pause;
end




