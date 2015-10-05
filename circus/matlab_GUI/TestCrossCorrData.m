clear all
close all

BinSize = 20;%1 ms

MaxDelay = 100;% in BinSize

overlapThres = 0.75;

Display = 0;

filename = '/Users/olivier/ownCloud/SpikeSorting/Allen/silico_0/silico_01';
% filename = '~/ownCloud/SpikeSorting/kenneth/20141202_all';
Extension = 'CC';

a = load([filename '.templates.mat'],'-mat');
templates = a.templates;

a = load([filename '.overlap.mat'],'-mat');

if isfield(a,'maxoverlap')
    overlap = a.maxoverlap / (size(templates,1)*size(templates,2));
else
    if ndims(a.overlap)==3
        overlap = max(a.overlap,[],3) / (size(templates,1)*size(templates,2));
    else
        overlap = a.overlap;
    end
end

NbTemplates = size(templates,3)/2;

a = load([filename '.spiketimes.mat'],'-mat');
if isfield(a,'temp_0')
    for i=1:NbTemplates
        SpikeTimes{i} = double(eval(['a.temp_' int2str(i-1)]));
    end
else
    SpikeTimes = a.SpikeTimes;
end



if Display>0
    figure;
end

Delays = (-MaxDelay:MaxDelay);

ListOverlap = [];
ListDiffCorr = [];
ListDiffCorrNorm = [];
IndexI = [];
IndexJ = [];
ListMean = [];

DelayAverage = 2;%In BinSize
ToAverage = find(Delays>=-DelayAverage & Delays<=DelayAverage );

for i=1:NbTemplates
    i
    t1 = double(SpikeTimes{i});
    if ~isempty(t1)
        t1b = floor(t1/BinSize);
        t1b = unique(t1b);

        count = 0;
        for j=i+1:NbTemplates
            if overlap(i,j)>overlapThres
                t2 = double(SpikeTimes{j});

                if ~isempty(t2)
                    
                    ListOverlap = [ListOverlap ; overlap(i,j)];
                    
                    t2b = floor(t2/BinSize);

                    t2b = unique(t2b);

                    t2bInverted = t2b(end) + t2b(1) - t2b;

                    CorrCount(:,i,j) = ones(2*MaxDelay+1,1,1)*(length(t1b) + length(t2b));
                    CorrCountInverted(:,i,j) = CorrCount(:,i,j);

                    for d=1:(2*MaxDelay+1)
                        t2bShifted = t2b + (d - MaxDelay - 1);
                        CorrCount(d,i,j) = CorrCount(d,i,j) - length(unique([t1b(:) ; t2bShifted(:)]));

                        t2bInvertedShifted = t2bInverted + (d - MaxDelay - 1);
                        CorrCountInverted(d,i,j) = CorrCountInverted(d,i,j) - length(unique([t1b(:) ; t2bInvertedShifted(:)]));
                    end

                    ListDiffCorr = [ListDiffCorr ; (mean(squeeze(CorrCount(ToAverage,i,j))) - mean(squeeze(CorrCountInverted(ToAverage,i,j)))) ];
                    ListDiffCorrNorm = [ListDiffCorrNorm ; (mean(squeeze(CorrCount(ToAverage,i,j))) - mean(squeeze(CorrCountInverted(ToAverage,i,j))))/mean(squeeze(CorrCountInverted(ToAverage,i,j))) ];
                    IndexI = [IndexI ; i ];
                    IndexJ = [IndexJ ; j ];
                    ListMean = [ListMean ; mean(squeeze(CorrCountInverted(ToAverage,i,j)))];
                    DiffC(i,j) = mean(squeeze(CorrCount(ToAverage,i,j))) - mean(squeeze(CorrCountInverted(ToAverage,i,j)));
                    
                    count = count + 1;

                    if Display>0
                        plot(Delays,squeeze(CorrCount(:,i,j)))
                        hold on
                        plot(Delays,squeeze(CorrCountInverted(:,i,j)),'r')
                        hold off
                        title([int2str(i) ' ' int2str(j)])
                        pause
                    end
                end
            end
        end
    end
end

save([filename '.' Extension '.mat'],'ListOverlap','ListDiffCorr','ListDiffCorrNorm',...
    'ListMean','CorrCount','CorrCountInverted','IndexI','IndexJ',...
    'BinSize','MaxDelay','overlapThres','-mat','-v7.3');

    
