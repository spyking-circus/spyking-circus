clear all
close all

% filename = '~/ownCloud/SpikeSorting/kenneth/20141202_all';
filename = '/Users/olivier/ownCloud/SpikeSorting/Allen/silico_0/silico_01';

Extension = 'CC';

% BinSize = 20;%1 ms

% MaxDelay = 100;% in BinSize

load([filename '.' Extension '.mat'],'-mat')

Delays = (-MaxDelay:MaxDelay);

MinOverlap = min(ListOverlap);

% figure;
% plot(ListOverlap,ListDiffCorr,'.')
% 
% figure;
% plot(ListOverlap,ListDiffCorrNorm,'.')
% 
% figure;
% plot(ListDiffCorr,ListDiffCorrNorm,'.')

ToMergeAll = zeros(1,length(ListOverlap));

w = 1;

while w>0

    subs = find(ListDiffCorr<2);
    
    subsMerge = find(ToMergeAll>0);
    
    figure(1);
    plot(ListMean,ListDiffCorr,'.')
    if ~isempty(subsMerge)
        hold on
        plot(ListMean(subsMerge),ListDiffCorr(subsMerge),'g.')
        hold off
    end
    set(gca,'FontSize',16)
    xlabel('Mean correlation ')
    ylabel('Correlation difference ')
    xlim([0 (max(ListMean(subs))+5)])
    ylim([(min(ListDiffCorr(subs))-5) 0])
    %Set lims for ListDiffCorr<0

    figure(2);
    plot(ListMean,ListOverlap,'.')
    if ~isempty(subsMerge)
        hold on
        plot(ListMean(subsMerge),ListOverlap(subsMerge),'g.')
        hold off
    end
    set(gca,'FontSize',16)
    xlabel('Mean correlation ')
    ylabel('Template similarity ')
    xlim([0 (max(ListMean(subs))+5)])
    ylim([MinOverlap 1])

    figure(3);
    plot(ListDiffCorr,ListOverlap,'.')
    if ~isempty(subsMerge)
        hold on
        plot(ListDiffCorr(subsMerge),ListOverlap(subsMerge),'g.')
        hold off
    end
    set(gca,'FontSize',16)
    xlabel('Correlation difference ')
    ylabel('Template similarity ')
    ylim([(min(ListDiffCorr)-5) 0])
    ylim([MinOverlap 1])


    figure(5);
    scatter3(ListMean(subs),ListDiffCorr(subs),ListOverlap(subs))
    if ~isempty(subsMerge)
        hold on
        scatter3(ListMean(subsMerge),ListDiffCorr(subsMerge),ListOverlap(subsMerge),'g')
        hold off
    end

    w = input('Pick the window you want to use, or 0 to exit:');
    %% Give the possibility to pick single cross-correlogram from points

    if w==1
        Y = ListDiffCorr;
        X = ListMean;
    end

    if w==2
        Y = ListOverlap;
        X = ListMean;
    end
    
    if w==3
        Y = ListOverlap;
        X = ListDiffCorr;
    end
    
    action = input('Pick a cross-correlogram (1), merge (2), or exit (0)');

%     if action==0
%         return
%     end
    
    if action == 1
        figure(w)

        [xp,yp] = ginput(1);

        xp = xp / max(X);

        X = X / max(X);

        d = (X - xp).^2 + (Y - yp).^2;

        [m,idp] = min(d);

        id = idp;
        % id = subs(idp);

        i = IndexI(id);
        j = IndexJ(id);

        figure(4);
        plot(Delays,squeeze(CorrCount(:,i,j)))
        hold on
        plot(Delays,squeeze(CorrCountInverted(:,i,j)),'r')
        hold off
        title(['Cross-corr between template ' int2str(i) ' and template ' int2str(j)])
        xlabel('Time ')
        
    end
    
    if action == 2
        
        figure(w)

        [xc,yc] = ginput;
        xc(length(xc)+1) = xc(1);
        yc(length(yc)+1) = yc(1);%To close the loop. 

        xc(length(xc)+1) = xc(2);
        yc(length(yc)+1) = yc(2);%To close the loop. 

        % % [X,Y,nonzero] = GetClusterValues(handles);

        a = ones(size(X));
        a = (a==1);
        %b = a;

        %This loop determined which points are inside the polygon,. Works if the
        %polygon is convex only!!!!!
        for i=1:(length(xc)-2)
            indexa = (((yc(i+1)-yc(i))/(xc(i+1)-xc(i))).*(X - xc(i)) + yc(i) - Y) * (((yc(i+1)-yc(i))/(xc(i+1)-xc(i))).*(xc(i+2) - xc(i)) + yc(i) - yc(i+2)) >0;%The points selected must be convex. 
            %indexb = ~indexa;

            a = a & indexa;
            %b = b & indexb;

        end
        % if sum(b)>sum(a)
        %     a=b;
        % end

        plot(X(a),Y(a),'r.')
        hold on
        plot(X(~a),Y(~a),'b.')
        hold off

        if w==1
            set(gca,'FontSize',16)
            xlabel('Mean correlation ')
            ylabel('Correlation difference ')
            xlim([0 (max(ListMean(subs))+5)])
            ylim([(min(ListDiffCorr(subs))-5) 0])
        end

        if w==2
            set(gca,'FontSize',16)
            xlabel('Mean correlation ')
            ylabel('Template similarity ')
            xlim([0 (max(ListMean(subs))+5)])
            ylim([MinOverlap 1])
        end

        if w==3
            set(gca,'FontSize',16)
            xlabel('Correlation difference ')
            ylabel('Template similarity ')
            ylim([(min(ListDiffCorr)-5) 0])
            ylim([MinOverlap 1])
        end

        choice = input('Merge red (1), blue (2), none (0):');

%         if choice==0
%             return
%         end

        if choice==1
            ToMergeAll(find(a>0)) = 1;
        end

        if choice==2
            ToMergeAll(find(a==0)) = 1;
        end

    end
    
end

iMerge = IndexI(find(ToMergeAll>0));
jMerge = IndexJ(find(ToMergeAll>0));

% All the selected pairs are saved in CC_merged.mat
MergedIndex = (1:size(CorrCount,3));

for k=length(jMerge):-1:1
    ValAtJ = find(MergedIndex==jMerge(k));
    MergedIndex(ValAtJ) = iMerge(k);
end

save([filename '.' Extension 'merged.mat'],'iMerge','jMerge','MergeIndex','ToMergeAll','-mat')


%% Create new data with everything merged

%Amplitudes

load([filename '.amplitudes.mat'],'-mat')

for i=1:length(Amplitudes)
    ValAtI = find(MergedIndex==i);
    NewAmplitudes{i} = [];
    NewAmplitudes2{i} = [];
    for k=1:length(ValAtI)
        NewAmplitudes{i} = [ NewAmplitudes{i}(:) ; Amplitudes{ValAtI(k)} ];
        NewAmplitudes2{i} = [ NewAmplitudes2{i}(:) ; Amplitudes2{ValAtI(k)} ];
    end
end

Amplitudes = NewAmplitudes;
Amplitudes2 = NewAmplitudes2;

save([filename '.amplitudes.merged.mat'],'Amplitudes','Amplitudes2','-mat')



%SpikeTimes

load([filename '.spiketimes.mat'],'-mat')

for i=1:length(SpikeTimes)
    ValAtI = find(MergedIndex==i);
    NewSpikeTimes{i} = [];
    for k=1:length(ValAtI)
        NewSpikeTimes{i} = [ NewSpikeTimes{i}(:) ; SpikeTimes{ValAtI(k)} ];
    end
end

SpikeTimes = NewSpikeTimes;

save([filename '.spiketimes.merged.mat'],'SpikeTimes','-mat')

%templates

load([filename '.spiketimes.mat'],'-mat')
for i=1:length(SpikeTimes)
    ValAtI = find(MergedIndex==i);
    NewSpikeTimes{i} = [];
    for k=1:length(ValAtI)
        NewSpikeTimes{i} = [ NewSpikeTimes{i}(:) ; SpikeTimes{ValAtI(k)} ];
    end
end

