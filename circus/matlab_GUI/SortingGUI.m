function varargout = SortingGUI(varargin)

% SortingGUI(20000,'/Users/olivier/ownCloud/SpikeSorting/kenneth/20141202_all/20141202_all','-final.mat','../mappings/kenneth.mapping.mat',2,'int16',0,0.1)
%SamplingRate, filename, extension, mappingfile, RPVlimit, format,
%HeaderSize, Gain

% SORTINGGUI MATLAB code for SortingGUI.fig
%      SORTINGGUI, by itself, creates a new SORTINGGUI or raises the existing
%      singleton*.
%
%      H = SORTINGGUI returns the handle to a new SORTINGGUI or the handle to
%      the existing singleton*.
%
%      SORTINGGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SORTINGGUI.M with the given input arguments.
%
%      SORTINGGUI('Property','Value',...) creates a new SORTINGGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SortingGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SortingGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SortingGUI

% Last Modified by GUIDE v2.5 03-Oct-2015 08:32:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SortingGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @SortingGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SortingGUI is made visible.
function SortingGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SortingGUI (see VARARGIN)

% Choose default command line output for SortingGUI
handles.output = hObject;

filename = varargin{2};
suffix = varargin{3};

if ~strcmp(suffix, '.mat')
    result = strsplit(suffix, '.mat');
    set(handles.VersionNb, 'String', result(1));
end

handles.SamplingRate = varargin{1};


if length(varargin)<=3
    load('../mappings/mea_252.mapping.mat','-mat')
    handles.Xmin = 1;
    handles.Xmax = 16;
    handles.Ymin = 1;
    handles.Ymax = 16;
else
    load(varargin{4},'-mat')
    handles.Xmin = min(Positions(:,1));
    handles.Xmax = max(Positions(:,1));
    handles.Ymin = min(Positions(:,2));
    handles.Ymax = max(Positions(:,2))+1;

end

handles.Positions = double(Positions);

if length(varargin)<=4
    handles.RPVlim = 2;
else
    handles.RPVlim = varargin{5};
end


handles.filename = filename;


%% Template file: could also contain AmpLim and AmpTrend

if exist([filename '.templates' suffix '-mat'])
    template = load([filename '.templates' suffix],'-mat');
    handles.templates  = template.templates(:,:,1:end/2);
    handles.templates2 = template.templates(:,:,end/2+1:end);
else
    tmpfile = [filename '.templates' suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    template = h5read(tmpfile, '/templates');
    handles.templates  = template(:,:,1:end/2);
    handles.templates2 = template(:,:,end/2+1:end);
end

%% Raw data file if it is there

if length(varargin)>=6 
    data_file = varargin{9};
    if (exist(data_file,'file'))
        handles.DataFormat = varargin{6};
        handles.HeaderSize = varargin{7};
        handles.Gain       = varargin{8};
        handles.DataFid    = fopen(data_file,'r');
        
        if handles.HeaderSize<0%Read the MCS header automatically
            HeaderText = '';
            stop=0;
            HeaderSize=0;

            while (stop==0)&&(HeaderSize<=10000)%to avoid infinite loop
                ch = fread(handles.DataFid, 1, 'uint8=>char')';
                HeaderSize = HeaderSize + 1;
                HeaderText = [HeaderText ch];
                if (HeaderSize>2)
                    if strcmp(HeaderText((HeaderSize-2):HeaderSize),'EOH')
                        stop = 1;
                    end
                end
            end

            HeaderSize = HeaderSize+2;%Because there is two characters after the EOH, before the raw data. 

            handles.HeaderSize = HeaderSize;
        end

        handles.DataStartPt = 1000;

        if exist([filename '.whitening' suffix '-mat'])
            a = load([filename '.whitening.mat'],'-mat');
            handles.WhiteSpatial = a.spatial;
            handles.WhiteTemporal = a.temporal;
        else
            tmpfile = [filename '.basis' '.hdf5'];
            handles.WhiteSpatial = h5read(tmpfile, '/spatial');
            handles.WhiteTemporal = h5read(tmpfile, '/temporal');
        end

        b = load(varargin{4},'-mat');

        if isfield(b,'n_total')
            handles.NelecTot = b.n_total;
            for i=1:length(b.permutation)
                handles.ElecPermut(i) = b.permutation{i};
            end
        else
            handles.NelecTot = size(handles.templates,1);
            handles.ElecPermut = (0:size(handles.templates,1)-1);
        end
    end
end



%% Tagged

if isfield(template,'Tagged')
    handles.Tagged = template.Tagged;
else
    handles.Tagged = zeros(size(handles.templates,3),1);
end

%% spiketimes file
if exist([filename '.spiketimes' suffix],'file')    
    a = load([filename '.spiketimes' suffix],'-mat');
    if isfield(a,'SpikeTimes')
        handles.SpikeTimes = a.SpikeTimes;
        for id=1:size(handles.templates,3)
            handles.SpikeTimes{id} = handles.SpikeTimes{id}(:)/(handles.SamplingRate/1000);
        end
    else
        for id=1:size(handles.templates,3)
            handles.SpikeTimes{id} = double(eval(['a.temp_' int2str(id-1)]))/(handles.SamplingRate/1000);
            handles.SpikeTimes{id} = handles.SpikeTimes{id}(:);
        end
    end
else
    if ~isfield(handles,'SpikeTimes')
        handles.SpikeTimes = cell(1,size(handles.templates,3));
    end
end

%% Amplitude Limits

if exist([filename '.limits' suffix],'file')
    b = load([filename '.limits' suffix],'-mat');
    handles.AmpLim = b.limits;
else
    tmpfile = [filename '.templates' suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    handles.AmpLim = h5read(tmpfile, '/limits');
end

if ~isfield(template,'AmpTrend') || exist([filename '.limits' suffix],'file')
    
    m = 0;
    for i=1:length(handles.SpikeTimes)
        if size(handles.SpikeTimes{i}, 1) > 1
            m = max(m,max(handles.SpikeTimes{i}));   
        end
    end
    
    AmpTrend = cell(1,size(handles.templates,3));

    for i=1:length(handles.SpikeTimes)
        AmpTrend{i}([1 2],1) = [0 m];
        AmpTrend{i}([1 2],2) = [1 1];
    end
    
    handles.AmpTrend = AmpTrend;
else
    handles.AmpTrend = template.AmpTrend;
end

%% stimulus file: check if it exists

if exist([filename '.stim'],'file')
    a = load([filename '.stim'],'-mat');
    handles.StimBeg = a.rep_begin_time;
    handles.StimEnd = a.rep_end_time;
end

%% Amplitudes file: does not have to exist

if exist([filename '.amplitudes' suffix],'file') 
    a = load([filename '.amplitudes' suffix],'-mat');
    
    if isfield(a,'Amplitudes')
        handles.Amplitudes = a.Amplitudes;
        if isfield(a,'Amplitudes2')
            handles.Amplitudes2 = a.Amplitudes2;
        end
    else
        for id=1:size(handles.templates,3)
            if ~isempty(eval(['a.temp_' int2str(id-1)]))
                handles.Amplitudes{id} = double(eval(['a.temp_' int2str(id-1) '(:,1)']));
                handles.Amplitudes2{id} = double(eval(['a.temp_' int2str(id-1) '(:,2)']));
                
                handles.Amplitudes{id} = handles.Amplitudes{id}(:);
                handles.Amplitudes2{id} = handles.Amplitudes2{id}(:);
            else
                handles.Amplitudes{id} = [];
                handles.Amplitudes2{id} = [];
            end
        end
    end
else
    if exist([filename '.raster' suffix],'file')  
        a = load([filename '.raster' suffix],'-mat');
        handles.Amplitudes = a.Amplitudes;
        handles.SpikeTimes = a.SpikeTimes;
        for i=1:length(handles.SpikeTimes)
            handles.SpikeTimes{i} = double(handles.SpikeTimes{i})/(handles.SamplingRate/1000);
        end
    else
        
        handles.Amplitudes = cell(1,size(handles.templates,3));
        handles.Amplitudes2 = cell(1,size(handles.templates,3));
    end
end


%% Clusters file

if exist([filename '.clusters' suffix], 'file')    
    a = load([filename '.clusters' suffix],'-mat');
    if isfield(a,'clusters') %This is the save format
        handles.clusters     = a.clusters;
        handles.DistribClust = a.DistribClust;
        handles.BestElec     = a.BestElec;
    else 
        handles.BestElec = a.electrodes + 1;
        for id=1:size(handles.templates,1)%number of electrodes
            if ~isempty(eval(['a.clusters_' int2str(id-1)]))
                features = double(eval(['a.data_' int2str(id-1)]));
                ClusterId = double(eval(['a.clusters_' int2str(id-1)]));

                Values = sort(unique(ClusterId),'ascend');
                if Values(1)==-1
                    Values(1) = [];
                end
                
                corresponding_template_nbs = find(a.electrodes==id-1);
                if length(corresponding_template_nbs) > 0
                    for idx=1:length(Values)
                        sf = features(find(ClusterId==Values(idx)),:);
                        handles.clusters{corresponding_template_nbs(idx)} = sf;
                    end
                end
                handles.DistribClust{id} = eval(['a.debug_' int2str(id-1)]);
            end
        end
    end
else
    tmpfile = [filename '.clusters' suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    handles.BestElec = h5read(tmpfile, '/electrodes') + 1;
    for id=1:size(handles.templates,1)%number of electrodes
        if ~isempty(h5read(tmpfile, ['/clusters_' int2str(id-1)]))
            features  = h5read(tmpfile, ['/data_' int2str(id-1)]);
            ClusterId = h5read(tmpfile, ['/clusters_' int2str(id-1)]);

            Values = sort(unique(ClusterId),'ascend');
            if Values(1)==-1
                Values(1) = [];
            end
            
            corresponding_template_nbs = find(handles.BestElec==id);
            if length(corresponding_template_nbs) > 0
                for idx=1:length(Values)
                    sf = features(find(ClusterId==Values(idx)),:);
                    handles.clusters{corresponding_template_nbs(idx)} = sf;
                end
            end
            handles.DistribClust{id} = h5read(tmpfile, ['/debug_' int2str(id-1)]);
        end
    end
end

handles.Clims = zeros(size(handles.templates,1),2);
for i=1:length(handles.DistribClust)
    if ~isempty(handles.DistribClust{i})
        handles.Clims(i,1) = max(handles.DistribClust{i}(:,1));
        handles.Clims(i,2) = max(handles.DistribClust{i}(:,2));
    end
end


%% overlap

if exist([filename '.overlap' suffix], 'file')
    a               = load([filename '.overlap' suffix],'-mat');
    handles.overlap = a.maxoverlap/(size(handles.templates,1) * size(handles.templates,2));
else
    tmpfile = [filename '.overlap' suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    handles.overlap = h5read(tmpfile, '/maxoverlap')/(size(handles.templates,1) * size(handles.templates,2));
end

if size(handles.overlap, 1) == size(handles.templates,3)*2
    handles.overlap = handles.overlap(1:end/2,1:end/2,:);
end

handles.TemplateDisplayRatio = 0.9;



% % %% Sort the templates from the highest to the lowest and reorder everything
% % 
% % m = max(max(abs(handles.templates)));
% % 
% % [v,id] = sort(m,'descend');
% % 
% % handles.templates = handles.templates(:,:,id);
% % handles.templates2 = handles.templates2(:,:,id);
% % 
% % handles.overlap = handles.overlap(id,id,:);
% % 
% % handles.Amplitudes = handles.Amplitudes(id);
% % handles.Amplitudes2 = handles.Amplitudes2(id);
% % 
% % handles.AmpLim = handles.AmpLim(id,:);
% % handles.AmpTrend = handles.AmpTrend(id);
% % handles.Tagged = handles.Tagged(id);
% % 
% % handles.SpikeTimes = handles.SpikeTimes(id);
% % 
% % handles.clusters = handles.clusters(id);
% % handles.BestElec = handles.BestElec(id);

%% Plot

PlotData(handles)

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SortingGUI wait for user response (see UIRESUME)
% uiwait(handles.SortingGUI);


% --- Outputs from this function are returned to the command line.
function varargout = SortingGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in ZoomInBtn.
function ZoomInBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ZoomInBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%Find electrode with max for template
CellNb = str2num(get(handles.TemplateNb,'String'));
template(:,:) = handles.templates(:,:,CellNb);
m = max(abs(template),[],2);
[m,elec] = max(m);

Mx = handles.Positions(elec,1);
My = handles.Positions(elec,2);

if handles.Xmax > handles.Xmin%otherwise we don't change anything - only 1 column of electrodes
    if (Mx-handles.Xmin) > (handles.Xmax-Mx)
        handles.Xmin = handles.Xmin + 1;
    else 
        handles.Xmax = handles.Xmax - 1;
    end
end


if handles.Ymax > handles.Ymin%otherwise we don't change anything - only 1 line of electrodes
    if (My-handles.Ymin) > (handles.Ymax-My)
        handles.Ymin = handles.Ymin + 1;
    else 
        handles.Ymax = handles.Ymax - 1;
    end
end

%
% 
% DmaxX = max(Mx-handles.Xmin,handles.Xmax-Mx);
% RatioX = (DmaxX-1)/DmaxX;
% Xmin = round(Mx - RatioX*(Mx-handles.Xmin));
% Xmax = round(Mx + RatioX*(handles.Xmax-Mx));
% if Xmin<Xmax
%     handles.Xmin = Xmin;
%     handles.Xmax = Xmax;
% end
% 
% DmaxY = max(My-handles.Ymin,handles.Ymax-My);
% RatioY = (DmaxY-1)/DmaxY;
% Ymin = round(My - RatioY*(My-handles.Ymin));
% Ymax = round(My + RatioY*(handles.Ymax-My));
% if Ymin<Ymax
%     handles.Ymin = Ymin;
%     handles.Ymax = Ymax;
% end

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in ZoonOutBtn.
function ZoonOutBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ZoonOutBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


%Find electrode with max for template
CellNb = str2num(get(handles.TemplateNb,'String'));
template(:,:) = handles.templates(:,:,CellNb);
m = max(abs(template),[],2);
[m,elec] = max(m);

Mx = handles.Positions(elec,1);
My = handles.Positions(elec,2);


if (Mx-handles.Xmin) < (handles.Xmax-Mx) & handles.Xmin>min(handles.Positions(:,1))
    handles.Xmin = handles.Xmin - 1;
else 
    if handles.Xmax<max(handles.Positions(:,1))
        handles.Xmax = handles.Xmax + 1;
    else
        if handles.Xmin>min(handles.Positions(:,1))
            handles.Xmin = handles.Xmin - 1;
        end
    end
end

if (My-handles.Ymin) < (handles.Ymax-My) & handles.Ymin>min(handles.Positions(:,2))
    handles.Ymin = handles.Ymin - 1;
else 
    if handles.Ymax<max(handles.Positions(:,2))
        handles.Ymax = handles.Ymax + 1;
    else
        if handles.Ymin>min(handles.Positions(:,2))
            handles.Ymin = handles.Ymin - 1;
        end
    end
end

% DminX = max(Mx-handles.Xmin,handles.Xmax-Mx);
% RatioX = (DminX+1)/DminX;
% Xmin = round(Mx - RatioX*(Mx-handles.Xmin));
% Xmax = round(Mx + RatioX*(handles.Xmax-Mx));
% if Xmin>=1
%     handles.Xmin = Xmin;
% end
% 
% if Xmax<=16
%     handles.Xmax = Xmax;
% end
% 
% DminY = max(My-handles.Ymin,handles.Ymax-My);
% RatioY = (DminY+1)/DminY;
% Ymin = round(My - RatioY*(My-handles.Ymin));
% Ymax = round(My + RatioY*(handles.Ymax-My));

% if Ymin<=min(Positions(:,2))
%     handles.Ymin = Ymin;
% end
% 
% if Ymax>=max(Positions(:,2))
%     handles.Ymax = Ymax;
% end

guidata(hObject, handles);

PlotData(handles)



function FeatureY_Callback(hObject, eventdata, handles)
% hObject    handle to FeatureY (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FeatureY as text
%        str2double(get(hObject,'String')) returns contents of FeatureY as a double
PlotData(handles)


% --- Executes during object creation, after setting all properties.
function FeatureY_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FeatureY (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function FeatureX_Callback(hObject, eventdata, handles)
% hObject    handle to FeatureX (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FeatureX as text
%        str2double(get(hObject,'String')) returns contents of FeatureX as a double
PlotData(handles)


% --- Executes during object creation, after setting all properties.
function FeatureX_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FeatureX (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in FeatureYplus.
function FeatureYplus_Callback(hObject, eventdata, handles)
% hObject    handle to FeatureYplus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


CellNb = str2num(get(handles.TemplateNb,'String'));

FeatureY = str2num(get(handles.FeatureY,'String'));
if FeatureY < size(handles.clusters{CellNb},2)
    FeatureY = FeatureY + 1;
end

set(handles.FeatureY,'String',int2str(FeatureY))

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in FeatureYminus.
function FeatureYminus_Callback(hObject, eventdata, handles)
% hObject    handle to FeatureYminus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


FeatureY = str2num(get(handles.FeatureY,'String'));
if FeatureY >1
    FeatureY = FeatureY - 1;
end

set(handles.FeatureY,'String',int2str(FeatureY))

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in FeatureXplus.
function FeatureXplus_Callback(hObject, eventdata, handles)
% hObject    handle to FeatureXplus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));

FeatureX = str2num(get(handles.FeatureX,'String'));
if FeatureX < size(handles.clusters{CellNb},2)
    FeatureX = FeatureX + 1;
end

set(handles.FeatureX,'String',int2str(FeatureX))

guidata(hObject, handles);

PlotData(handles)


% --- Executes on button press in FeatureXminus.
function FeatureXminus_Callback(hObject, eventdata, handles)
% hObject    handle to FeatureXminus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

FeatureX = str2num(get(handles.FeatureX,'String'));
if FeatureX >1
    FeatureX = FeatureX - 1;
end

set(handles.FeatureX,'String',int2str(FeatureX))

guidata(hObject, handles);

PlotData(handles)


function TemplateNb_Callback(hObject, eventdata, handles)
% hObject    handle to TemplateNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TemplateNb as text
%        str2double(get(hObject,'String')) returns contents of TemplateNb as a double

PlotData(handles)

% --- Executes during object creation, after setting all properties.
function TemplateNb_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TemplateNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

ClearCrossCorr(hObject, eventdata, handles);


function Template2Nb_Callback(hObject, eventdata, handles)
% hObject    handle to Template2Nb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Template2Nb as text
%        str2double(get(hObject,'String')) returns contents of Template2Nb as a double

ClearCrossCorr(hObject, eventdata, handles);

PlotData(handles)

% --- Executes during object creation, after setting all properties.
function Template2Nb_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Template2Nb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in TemplateNbPlus.
function TemplateNbPlus_Callback(hObject, eventdata, handles)
% hObject    handle to TemplateNbPlus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));
if CellNb < size(handles.templates,3)
    CellNb = CellNb + 1;
end
set(handles.TemplateNb,'String',int2str(CellNb))


ClearCrossCorr(hObject, eventdata, handles);

ViewMode = str2num(get(handles.SwitchViewNb,'String'));

if ViewMode==1
    ViewMode = 3 - str2num(get(handles.SwitchViewNb,'String'));
    set(handles.SwitchViewNb,'String',int2str(ViewMode));

end

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in TemplateNbMinus.
function TemplateNbMinus_Callback(hObject, eventdata, handles)
% hObject    handle to TemplateNbMinus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));
if CellNb > 1
    CellNb = CellNb - 1;
end
set(handles.TemplateNb,'String',int2str(CellNb))

ClearCrossCorr(hObject, eventdata, handles);

ViewMode = str2num(get(handles.SwitchViewNb,'String'));

if ViewMode==1
    ViewMode = 3 - str2num(get(handles.SwitchViewNb,'String'));
    set(handles.SwitchViewNb,'String',int2str(ViewMode));

end

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in Template2NbMinus.
function Template2NbMinus_Callback(hObject, eventdata, handles)
% hObject    handle to Template2NbMinus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb2 = str2num(get(handles.Template2Nb,'String'));
if CellNb2 > 1
    CellNb2 = CellNb2 - 1;
end
set(handles.Template2Nb,'String',int2str(CellNb2))

ClearCrossCorr(hObject, eventdata, handles);

guidata(hObject, handles);

PlotData(handles)


% --- Executes on button press in Template2NbPlus.
function Template2NbPlus_Callback(hObject, eventdata, handles)
% hObject    handle to Template2NbPlus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb2 = str2num(get(handles.Template2Nb,'String'));
if CellNb2 < size(handles.templates,3)
    CellNb2 = CellNb2 + 1;
end
set(handles.Template2Nb,'String',int2str(CellNb2))

ClearCrossCorr(hObject, eventdata, handles);

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in CellGrade.
function CellGrade_Callback(hObject, eventdata, handles)
% hObject    handle to CellGrade (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));

if handles.Tagged(CellNb)<5
    handles.Tagged(CellNb) = handles.Tagged(CellNb) + 1;
else
    handles.Tagged(CellNb) = 0;
end

guidata(hObject, handles);

GradeStr{1} = 'O';
GradeStr{2} = 'E';
GradeStr{3} = 'D';
GradeStr{4} = 'C';
GradeStr{5} = 'B';
GradeStr{6} = 'A';

set(handles.CellGrade,'String',GradeStr{handles.Tagged(CellNb)+1});



% --- Executes on button press in SwitchViewNb.
function SwitchViewNb_Callback(hObject, eventdata, handles)
% hObject    handle to SwitchViewNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ViewMode = 3 - str2num(get(handles.SwitchViewNb,'String'));
set(handles.SwitchViewNb,'String',int2str(ViewMode));

guidata(hObject, handles);

PlotData(handles)


% --- Executes on button press in SuggestSimilar.
function SuggestSimilar_Callback(hObject, eventdata, handles)
% hObject    handle to SuggestSimilar (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

SimilarNb = str2num(get(handles.SimilarNb,'String'));

CellNb = str2num(get(handles.TemplateNb,'String'));
comp   = squeeze(handles.overlap(CellNb,:));

if get(handles.SameElec,'Value')~=0
    comp(find(handles.BestElec ~= handles.BestElec(CellNb))) = 0;
end

comp     = max(comp,[],1);
[val,id] = sort(comp,'descend');
IdTempl  = id(SimilarNb+1);%The first one is the template with itself. 

if get(handles.SameElec,'Value')~=0 & val(SimilarNb+1)==0
    disp('No more templates to compare in the same electrode')
end


set(handles.Template2Nb,'String',int2str(IdTempl))

set(handles.SwitchViewNb,'String','1')

if get(handles.SameElec,'Value')~=0 && val(SimilarNb+1)~=0 %We can compare the cluster
    mf1 = median(handles.clusters{CellNb});
    mf2 = median(handles.clusters{IdTempl});

    [m,idf] = sort(abs(mf1-mf2),'descend');
    
    set(handles.FeatureX,'String',int2str(idf(1)));
    set(handles.FeatureY,'String',int2str(idf(2)));

end

guidata(hObject, handles);

PlotData(handles)


function SimilarNb_Callback(hObject, eventdata, handles)
% hObject    handle to SimilarNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of SimilarNb as text
%        str2double(get(hObject,'String')) returns contents of SimilarNb as a double

SuggestSimilar_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function SimilarNb_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SimilarNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in MergeTemplates.
function MergeTemplates_Callback(hObject, eventdata, handles)
% hObject    handle to MergeTemplates (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%We keep the data from template 1, merge what can be merged from template 2
%and delete the rest. 

CellNb = str2num(get(handles.TemplateNb,'String'));
CellNb2 = str2num(get(handles.Template2Nb,'String'));

t = [handles.SpikeTimes{CellNb}(:) ; handles.SpikeTimes{CellNb2}(:) ];
[t,id] = sort(t,'ascend');

a = [handles.Amplitudes{CellNb}(:) ; handles.Amplitudes{CellNb2}(:) ];
a = a(id);

handles.SpikeTimes{CellNb} = t;
handles.Amplitudes{CellNb} = a;

handles.SpikeTimes(CellNb2) = [];
handles.Amplitudes(CellNb2) = [];
handles.Amplitudes2(CellNb2) = [];

handles.templates(:,:,CellNb2) = [];
handles.templates2(:,:,CellNb2) = [];
handles.AmpLim(CellNb2,:) = [];
handles.AmpTrend(CellNb2) = [];
handles.clusters(CellNb2) = [];
handles.BestElec(CellNb2) = [];
% handles.DistribClust(CellNb2) = [];
handles.Tagged(CellNb2) = [];


handles.overlap(CellNb2,:) = [];
handles.overlap(:,CellNb2) = [];

if CellNb2<CellNb
    set(handles.TemplateNb,'String',int2str(CellNb-1));
end

guidata(hObject, handles);

PlotData(handles)

% --- Executes on button press in ManualTrend.
function ManualTrend_Callback(hObject, eventdata, handles)
% hObject    handle to ManualTrend (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


CellNb = str2num(get(handles.TemplateNb,'String'));

axes(handles.AmpTimeWin);

[xc,yc] = ginput;

%add points at the beginning and at the end if necessary

xc=round(xc);
if xc>1 
    xc = [1 ; xc];
    yc = [yc(1) ; yc];
end

if xc(end) < max(handles.SpikeTimes{CellNb})
    xc = [xc ; max(handles.SpikeTimes{CellNb})];
    yc = [yc ; yc(end)];
end

tr(:,1) = xc';
tr(:,2) = yc';

handles.AmpTrend{CellNb} = tr;

guidata(hObject, handles);

PlotData(handles)


function TrendSliceNb_Callback(hObject, eventdata, handles)
% hObject    handle to TrendSliceNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TrendSliceNb as text
%        str2double(get(hObject,'String')) returns contents of TrendSliceNb as a double

CellNb = str2num(get(handles.TemplateNb,'String'));

TrendSliceNb = str2num(get(handles.TrendSliceNb,'String'));

x = linspace(0,max(handles.SpikeTimes{CellNb}),TrendSliceNb);

r = handles.SpikeTimes{CellNb}(:);

for i=1:length(x)
    if i<length(x)
        
        SubAmps = handles.Amplitudes{CellNb}( find( r >= x(i) & r<= x(i+1)) );
        y(i) = median(SubAmps);
    else 
        y(i) = y(i-1);
    end
end


tr(:,1) = x';
tr(:,2) = y';

handles.AmpTrend{CellNb} = tr;

guidata(hObject, handles);

PlotData(handles)


% --- Executes during object creation, after setting all properties.
function TrendSliceNb_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TrendSliceNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in DefineTrend.
function DefineTrend_Callback(hObject, eventdata, handles)
% hObject    handle to DefineTrend (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


TrendSliceNb_Callback(hObject, eventdata, handles);



% --- Executes on button press in SetAmpMin.
function SetAmpMin_Callback(hObject, eventdata, handles)
% hObject    handle to SetAmpMin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));

%This is to determine amplitude min. 
axes(handles.AmpTimeWin);
[x,y] = ginput(1);

Trend = handles.AmpTrend{CellNb};


SubTrend = interp1(Trend(:,1),Trend(:,2),x);
Lim = y./SubTrend;
handles.AmpLim(CellNb,1) = Lim;

guidata(hObject, handles);

PlotData(handles)

% % % template = squeeze(handles.templates(:,:,CellNb));
% % % 
% % % m = max(abs(template),[],2);
% % % 
% % % [mm,LargestElec] = max(m);
% % % 
% % % a = handles.Amplitudes{CellNb};
% % % t = handles.SpikeTimes{CellNb};
% % % V=[];
% % % P=[];
% % % 
% % % duration = size(handles.templates,2);
% % % 
% % % if duration/2 == round(duration/2)
% % %     duration = duration + 1;
% % % end
% % % 
% % % for ispike=1:length(t)
% % %     tstart = double(handles.SpikeTimes{CellNb}(ispike)*(handles.SamplingRate/1000))  - (duration+1)/2 + 1 ;
% % % 
% % %     NbElec = handles.NelecTot;
% % % 
% % %     FileStart = handles.HeaderSize + 2*NbElec*tstart;%We assume that each voltage value is written on 2 bytes. Otherwise this line must be changed. 
% % % 
% % % 
% % %     FullStart  = FileStart - size(handles.templates,2)*NbElec*2;
% % %     FullLength = (duration + 2*size(handles.templates,2))*NbElec;
% % % 
% % %     fseek(handles.DataFid,FullStart,'bof');
% % % 
% % %     data = double(fread(handles.DataFid,FullLength,handles.DataFormat));
% % % 
% % %     if strcmp(handles.DataFormat,'uint16')
% % %         data = data - 32767;
% % %     end
% % % 
% % %     data = data*handles.Gain;
% % % 
% % %     data = reshape(data,[NbElec (duration + 2*size(handles.templates,2))]);
% % % 
% % %     %% Filtering
% % % 
% % %     data = data(handles.ElecPermut + 1,:);
% % % 
% % %     data = handles.WhiteSpatial*data;
% % %     for i=1:size(data,1)
% % %         data(i,:) = conv(data(i,:),handles.WhiteTemporal,'same');
% % %     end
% % % 
% % %     %% Reduce the data to the portion of interest - remove also the unnecessary
% % %     %electrodes
% % %     RawData = data(:,(size(handles.templates,2)+1):(end-size(handles.templates,2)));
% % % 
% % %     %% Compare voltage value and the one from the template
% % % 
% % %     V(ispike) = RawData(LargestElec,(size(handles.templates,2)-1)/2+1);
% % %     P(ispike) = a(ispike)* template(LargestElec,(size(handles.templates,2)-1)/2+1);
% % % 
% % % end
% % % 
% % % figure;
% % % plot(V,P,'.')
% % % 
% % % figure;
% % % plot((V-P)./V,a,'.')
% % % 
% % % V

% --- Executes on button press in SetAmpMax.
function SetAmpMax_Callback(hObject, eventdata, handles)
% hObject    handle to SetAmpMax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));

%This is to determine amplitude min. 
axes(handles.AmpTimeWin);
[x,y] = ginput(1);

Trend = handles.AmpTrend{CellNb};

SubTrend = interp1(Trend(:,1),Trend(:,2),x);
Lim = y./SubTrend;
handles.AmpLim(CellNb,2) = Lim;

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in BigISI.
function BigISI_Callback(hObject, eventdata, handles)
% hObject    handle to BigISI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of BigISI

PlotData(handles)


function PlotData(handles)

CellNb = str2num(get(handles.TemplateNb,'String'));
CellNb2 = str2num(get(handles.Template2Nb,'String'));

set(handles.ElecNb,'String',int2str(handles.BestElec(CellNb)))
set(handles.ElecNb2,'String',int2str(handles.BestElec(CellNb2)))

ViewMode = 3 - str2num(get(handles.SwitchViewNb,'String'));

%% PLOT TEMPLATE
template = handles.templates(:,:,CellNb);
Yspacing = max(abs(template(:)));

GradeStr{1} = 'O';
GradeStr{2} = 'E';
GradeStr{3} = 'D';
GradeStr{4} = 'C';
GradeStr{5} = 'B';
GradeStr{6} = 'A';

set(handles.CellGrade,'String',GradeStr{handles.Tagged(CellNb)+1});

if ViewMode == 1
    
    if get(handles.EnableWaveforms,'Value')==0%Classical display with just the template
    
        if get(handles.NormalizeTempl,'Value')==0
            PlotWaveform(handles,template);
        else
            PlotWaveform(handles,template,Yspacing);
        end
    
    else
        
        %TemplateWindow = round(duration/2) + (-1*(size(template,2)-1)/2 : (size(template,2)-1)/2 );
        templateIn = handles.Amp2Fit*template;

        PlotWaveform(handles, handles.RawData, str2num(get(handles.Yspacing,'String')));
        hold(handles.TemplateWin,'on')
        
        PlotWaveform(handles, templateIn, str2num(get(handles.Yspacing,'String')),'r');
        hold(handles.TemplateWin,'off')
        
    end
    
else
    template2 = handles.templates(:,:,CellNb2);

    %Yspacing = max( max(abs(template(:))) , max(abs(template2(:))) );

    if get(handles.NormalizeTempl,'Value')==0
        PlotWaveform(handles,template,str2num(get(handles.Yspacing,'String')))
        hold(handles.TemplateWin,'on')
        PlotWaveform(handles,template2,str2num(get(handles.Yspacing,'String')),'r')
        hold(handles.TemplateWin,'off')
    else
        PlotWaveform(handles,template/max(abs(template(:))),1)
        hold(handles.TemplateWin,'on')
        PlotWaveform(handles,template2/max(abs(template2(:))),1,'r')
        hold(handles.TemplateWin,'off')
    end
    
    set(handles.SimilarityTemplates,'String',['Similarity: ' num2str(max(squeeze(handles.overlap(CellNb,CellNb2))) )])
    
%     ShowCorr_Callback;
end


%% TEMPLATE COUNT

set(handles.TemplateCountTxt,'String',[int2str(length(find(handles.Tagged>0))) '/' int2str(size(handles.templates,3))])


%% PLOT CLUSTER
FeatureX = str2num(get(handles.FeatureX,'String'));
FeatureY = str2num(get(handles.FeatureY,'String'));

plot(handles.ClusterWin,handles.clusters{CellNb}(:,FeatureX),handles.clusters{CellNb}(:,FeatureY),'.')

if (ViewMode==2) & (handles.BestElec(CellNb)==handles.BestElec(CellNb2))
    hold(handles.ClusterWin,'on')
    plot(handles.ClusterWin,handles.clusters{CellNb2}(:,FeatureX),handles.clusters{CellNb2}(:,FeatureY),'r.')
    hold(handles.ClusterWin,'off')
end


%% PLOT DistribClust
% % plot(handles.SeparationWin,handles.DistribClust{handles.BestElec(CellNb)}(:,1),handles.DistribClust{handles.BestElec(CellNb)}(:,2),'.')
% % 
% % rhomin = handles.ClusterLims(handles.BestElec(CellNb),1);
% % deltamin = handles.ClusterLims(handles.BestElec(CellNb),2);
% % 
% % Selec = find(handles.DistribClust{handles.BestElec(CellNb)}(:,1) >rhomin & handles.DistribClust{handles.BestElec(CellNb)}(:,2)>deltamin);
% % hold(handles.SeparationWin,'on')
% % plot(handles.SeparationWin,handles.DistribClust{handles.BestElec(CellNb)}(Selec,1),handles.DistribClust{handles.BestElec(CellNb)}(Selec,2),'r.')
% % hold(handles.SeparationWin,'off')
% % 
% % set(handles.SeparationWin,'Xlim',[0 handles.Clims(handles.BestElec(CellNb),1)])
% % set(handles.SeparationWin,'Ylim',[0 handles.Clims(handles.BestElec(CellNb),2)])

%% PLOT AMPLITUDE

if ~isempty(handles.SpikeTimes{CellNb})
    plot(handles.AmpTimeWin,handles.SpikeTimes{CellNb},handles.Amplitudes{CellNb},'.')
else
    plot(handles.AmpTimeWin,0,0,'.')
end
hold(handles.AmpTimeWin,'on')
plot(handles.AmpTimeWin,handles.AmpTrend{CellNb}(:,1),handles.AmpTrend{CellNb}(:,2),'color',[0.5 0.5 0.5],'LineWidth',2)
plot(handles.AmpTimeWin,handles.AmpTrend{CellNb}(:,1),handles.AmpTrend{CellNb}(:,2)*handles.AmpLim(CellNb,1),'color',[0.5 0.5 0.5],'LineWidth',2)
plot(handles.AmpTimeWin,handles.AmpTrend{CellNb}(:,1),handles.AmpTrend{CellNb}(:,2)*handles.AmpLim(CellNb,2),'color',[0.5 0.5 0.5],'LineWidth',2)

if (ViewMode==2) 
    plot(handles.AmpTimeWin,handles.SpikeTimes{CellNb2},handles.Amplitudes{CellNb2},'r.')
    
    t = [handles.SpikeTimes{CellNb}(:) ; handles.SpikeTimes{CellNb2}(:)];
    a = [handles.Amplitudes{CellNb}(:) ; handles.Amplitudes{CellNb2}(:)];
    
    [t,id] = sort(t,'ascend');
    a = a(id);
else
    t = handles.SpikeTimes{CellNb};
    a = handles.Amplitudes{CellNb};
end

ISI = diff(t);

RPV = find(ISI<handles.RPVlim);

if ~isempty(RPV)
    times_RPV(1,:) = t(RPV);
    times_RPV(2,:) = t(RPV+1);

    amp_RPV(1,:) = a(RPV);
    amp_RPV(2,:) = a(RPV+1);

    [max_amp,ind_max] = max(amp_RPV,[],1);
    max_times = times_RPV(2*(0:(size(times_RPV,2)-1))+ind_max);

    [min_amp,ind_min] = min(amp_RPV,[],1);
    min_times = times_RPV(2*(0:(size(times_RPV,2)-1))+ind_min);

    plot(handles.AmpTimeWin,max_times,max_amp,'y.')
    plot(handles.AmpTimeWin,min_times,min_amp,'g.')
end

if isfield(handles,'RawData')
    if (ViewMode==1) 
        if handles.AmpIndex<=length(handles.SpikeTimes{CellNb})
            plot(handles.AmpTimeWin,handles.SpikeTimes{CellNb}(handles.AmpIndex),handles.Amplitudes{CellNb}(handles.AmpIndex),'m.')
        end
    end
end

hold(handles.AmpTimeWin,'off')


%% PLOT ISI
if ViewMode==1
    t = handles.SpikeTimes{CellNb};
else
    t = [handles.SpikeTimes{CellNb}(:) ; handles.SpikeTimes{CellNb2}(:) ];
    t = sort(t,'ascend');
end

%Assuming t is in milliseconds
ISI = diff(t);
lenISI = length(ISI);
NbRPV = length(find(ISI<=handles.RPVlim));
RatioRPV = length(find(ISI<=handles.RPVlim))/lenISI;
if get(handles.BigISI,'Value')==0
    ISI = ISI(ISI<=25);
    hist(handles.ISIwin,ISI,100);
else
    ISI = ISI(ISI<=200);
    hist(handles.ISIwin,ISI,200);
end
% x = (0:2:26);
set(handles.RPV,'String',['RPV: ' num2str(RatioRPV*100) '%, ' int2str(NbRPV) '/' int2str(lenISI) ]);

%% PLOT RASTER

if isfield(handles,'StimBeg')

    cc = hsv(length(handles.StimBeg));

    for k=1:length(handles.StimBeg)
       ColorRaster{k} = cc(k,:);
    end

    hold(handles.RasterWin,'off');
    plot(handles.RasterWin,0,0);

    MaxLength = 0;
    MaxTimes = 0;

    LineCount = 1;
    for k=1:length(handles.StimBeg)
        fr_times = [];
        fr_repeat = [];
        
        plot(handles.RasterWin,[0 MaxTimes],[LineCount LineCount],'k');
        
        for i=1:length(handles.StimBeg{k})
            times = handles.SpikeTimes{CellNb}(handles.SpikeTimes{CellNb} >= handles.StimBeg{k}(i) & handles.SpikeTimes{CellNb} <= handles.StimEnd{k}(i)) ...
                - handles.StimBeg{k}(i);
            fr_times = [fr_times ; times(:)];
            fr_repeat = [fr_repeat ; LineCount*ones(length(times),1)];
            LineCount = LineCount + 1;
                        
        end
        
        LineCount = LineCount + 1;
        
%         fr_times = fr_times / (handles.SamplingRate/1000);
        
        if ~isempty(times)
            MaxTimes = max(MaxTimes,max(fr_times));
        end

        plot(handles.RasterWin,fr_times,fr_repeat,'.','color',ColorRaster{k});

        hold(handles.RasterWin,'on');
    end
    
    LineCount = 1;

    for k=1:length(handles.StimBeg)
        plot(handles.RasterWin,[0 MaxTimes],[LineCount LineCount],'k');
        
        LineCount = LineCount + length(handles.StimBeg{k});
    end
    
    hold(handles.RasterWin,'off');
end


function PlotWaveform(handles,waveform,Yspacing,wcolor,Width)
%Plot the waveform matrix.
%tstart and duration might be unused. 

if nargin<3
    Yspacing = str2num(get(handles.Yspacing,'String'));
%     Yspacing = max(abs(waveform(:)));
end

if nargin<4
    wcolor = [0 0 1];
end

if nargin<5
    Width=1;
end

Htemplate = handles.TemplateWin;

Xmin = handles.Xmin;
Ymin = handles.Ymin;
Xmax = handles.Xmax;
Ymax = handles.Ymax;

Coor = handles.Positions;

nch = size(handles.templates,1);

% % %DeNormalization
% % if get(handles.DeNormalization,'Value')>0
% %    waveform = handles.decov * waveform;
% % end

%Subselect the electrodes to be displayed: 

DisplayElec = find(Coor(:,1)>=Xmin & Coor(:,2)>=Ymin & Coor(:,1)<=Xmax & Coor(:,2)<=Ymax  );

handles.TemplateDisplayRatio = str2double(get(handles.Xscale,'String'));

for idElec=1:length(DisplayElec)%nch
    i = DisplayElec(idElec);
    X(:,i) = linspace(Coor(i,1),Coor(i,1)+handles.TemplateDisplayRatio,length(waveform(i,:)));
    Y(:,i) = Coor(i,2) + waveform(i,:)/Yspacing;
end

if ~isempty(X)
    plot(Htemplate,X,Y,'color',wcolor,'LineWidth',Width);

    set(Htemplate,'xlim',[double(Xmin-0.1) double(Xmax+1.1)],'ylim',[double(Ymin-1) double(Ymax+0.5)]);
    set(Htemplate,'xlim',[double(Xmin-0.1) double(Xmax+handles.TemplateDisplayRatio+0.1)],'ylim',[double(Ymin-1) double(Ymax+0.5)]);
end
set(Htemplate,'Xtick',[],'Ytick',[]);


% --- Executes on button press in KillBtn.
function KillBtn_Callback(hObject, eventdata, handles)
% hObject    handle to KillBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));

handles.SpikeTimes(CellNb) = [];
handles.Amplitudes(CellNb) = [];
handles.Amplitudes2(CellNb) = [];

handles.templates(:,:,CellNb) = [];
handles.templates2(:,:,CellNb) = [];
handles.AmpLim(CellNb,:) = [];
handles.AmpTrend(CellNb) = [];
handles.clusters(CellNb) = [];
handles.BestElec(CellNb) = [];
handles.Tagged(CellNb)   = [];

handles.overlap(CellNb,:) = [];
handles.overlap(:,CellNb) = [];

guidata(hObject, handles);

PlotData(handles)


function VersionNb_Callback(hObject, eventdata, handles)
% hObject    handle to VersionNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of VersionNb as text
%        str2double(get(hObject,'String')) returns contents of VersionNb as a double


% --- Executes during object creation, after setting all properties.
function VersionNb_CreateFcn(hObject, eventdata, handles)
% hObject    handle to VersionNb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in SaveBtn.
function SaveBtn_Callback(hObject, eventdata, handles)
% hObject    handle to SaveBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



%% Template file: could also contain AmpLim and AmpTrend

suffix = get(handles.VersionNb,'String')
filename = handles.filename;

templates = handles.templates;
templates(:,:,size(templates,3)+1:size(templates,3)*2) = handles.templates2;
AmpLim = handles.AmpLim;
AmpTrend = handles.AmpTrend;
Tagged = handles.Tagged;

save([filename '.templates' suffix '.mat'],'templates','AmpLim','AmpTrend','Tagged','-mat','-v7.3');


%% Amplitudes file: does not have to exist

Amplitudes = handles.Amplitudes;
Amplitudes2 = handles.Amplitudes2;

save([filename '.amplitudes' suffix '.mat'],'Amplitudes','Amplitudes2','-mat','-v7.3');

%% spiketimes file

SpikeTimes = handles.SpikeTimes;

for id=1:size(handles.templates,3)
    SpikeTimes{id} = SpikeTimes{id}(:)*(handles.SamplingRate/1000);
end

save([filename '.spiketimes' suffix '.mat'],'SpikeTimes','-mat','-v7.3');


%% Clusters file

clusters = handles.clusters;
DistribClust = handles.DistribClust;
BestElec = handles.BestElec;
ClusterLims = handles.ClusterLims;

save([filename '.clusters' suffix '.mat'],'clusters','DistribClust','BestElec','ClusterLims','-mat','-v7.3')

%% overlap

overlap = handles.overlap * (size(handles.templates,1) * size(handles.templates,2));

save([filename '.overlap' suffix '.mat'],'overlap','-mat','-v7.3');



% --- Executes on button press in SplitBtn.
function SplitBtn_Callback(hObject, eventdata, handles)
% hObject    handle to SplitBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%Duplicate the template

CellNb = str2num(get(handles.TemplateNb,'String'));

handles.SpikeTimes  = handles.SpikeTimes([(1:CellNb) (CellNb:length(handles.SpikeTimes))]);
handles.Amplitudes  = handles.Amplitudes([(1:CellNb) (CellNb:length(handles.Amplitudes))]);
handles.Amplitudes2 = handles.Amplitudes([(1:CellNb) (CellNb:length(handles.Amplitudes2))]);
handles.Tagged      = handles.Tagged([(1:CellNb) (CellNb:size(handles.templates,3))]);
handles.templates   = handles.templates(:,:,[(1:CellNb) (CellNb:size(handles.templates,3))]);
handles.templates2  = handles.templates2(:,:,[(1:CellNb) (CellNb:size(handles.templates2,3))]);

handles.AmpLim   = handles.AmpLim([(1:CellNb) (CellNb:size(handles.AmpLim,1))],:);
handles.AmpTrend = handles.AmpTrend([(1:CellNb) (CellNb:length(handles.AmpTrend))]);
handles.clusters = handles.clusters([(1:CellNb) (CellNb:length(handles.clusters))]);
handles.BestElec = handles.BestElec([(1:CellNb) (CellNb:length(handles.BestElec))]);

handles.overlap = handles.overlap([(1:CellNb) (CellNb:size(handles.overlap,1))],:);
handles.overlap = handles.overlap(:,[(1:CellNb) (CellNb:size(handles.overlap,2))]);

%Remove the amplitudes/spiketimes in or out of the amp lims

% handles.SpikeTimes = handles.SpikeTimes([(1:CellNb) (CellNb:length(handles.SpikeTimes))]);
% handles.Amplitudes = handles.Amplitudes([(1:CellNb) (CellNb:length(handles.Amplitudes))]);
% + Amplitudes2


Trend = interp1(handles.AmpTrend{CellNb}(:,1),handles.AmpTrend{CellNb}(:,2),handles.SpikeTimes{CellNb}(:));

ToKeep = Trend*handles.AmpLim(CellNb,1) <= handles.Amplitudes{CellNb} & Trend*handles.AmpLim(CellNb,2) >= handles.Amplitudes{CellNb};

handles.SpikeTimes{CellNb} = handles.SpikeTimes{CellNb}(find(ToKeep));
handles.Amplitudes{CellNb} = handles.Amplitudes{CellNb}(find(ToKeep));
handles.Amplitudes2{CellNb} = handles.Amplitudes2{CellNb}(find(ToKeep));

handles.SpikeTimes{CellNb+1} = handles.SpikeTimes{CellNb+1}(find(~ToKeep));
handles.Amplitudes{CellNb+1} = handles.Amplitudes{CellNb+1}(find(~ToKeep));
handles.Amplitudes2{CellNb+1} = handles.Amplitudes2{CellNb+1}(find(~ToKeep));

guidata(hObject, handles);

PlotData(handles)



% --- Executes on button press in SetClustLims.
function SetClustLims_Callback(hObject, eventdata, handles)
% hObject    handle to SetClustLims (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes(handles.SeparationWin);
[x,y] = ginput(1);

ElecNb = str2num(get(handles.ElecNb,'String'));

handles.ClusterLims(ElecNb,1) = x;
handles.ClusterLims(ElecNb,2) = y;
handles.ClusterLims(ElecNb,3) = 1;

guidata(hObject, handles);

PlotData(handles)


% --- Executes on button press in SameElec.
function SameElec_Callback(hObject, eventdata, handles)
% hObject    handle to SameElec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of SameElec


% --- Executes on button press in NormalizeTempl.
function NormalizeTempl_Callback(hObject, eventdata, handles)
% hObject    handle to NormalizeTempl (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of NormalizeTempl

PlotData(handles)


% --- Executes on button press in ZoomInClims.
function ZoomInClims_Callback(hObject, eventdata, handles)
% hObject    handle to ZoomInClims (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ElecNb = str2num(get(handles.ElecNb,'String'));

handles.Clims(ElecNb,:) = handles.Clims(ElecNb,:)/2;

guidata(hObject, handles);

PlotData(handles)

% --- Executes on button press in ZoomOutClims.
function ZoomOutClims_Callback(hObject, eventdata, handles)
% hObject    handle to ZoomOutClims (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ElecNb = str2num(get(handles.ElecNb,'String'));

handles.Clims(ElecNb,:) = handles.Clims(ElecNb,:)*2;

guidata(hObject, handles);

PlotData(handles)


% --- Executes on button press in ShowCorr.
function ShowCorr_Callback(hObject, eventdata, handles)
% hObject    handle to ShowCorr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));
CellNb2 = str2num(get(handles.Template2Nb,'String'));

ViewMode = 3 - str2num(get(handles.SwitchViewNb,'String'));


%% PLOT CROSS-CORR
% if 0%INSTALL GCC AND COMPILE crosscorrspike.cpp
if ViewMode==2
    t1 = handles.SpikeTimes{CellNb};
    t2 = handles.SpikeTimes{CellNb2};
    
    
    

    BinSize = 2;%1 ms

    MaxDelay = 100;% in BinSize

    %Here it starts

    t1b = floor(t1/BinSize);
    t2b = floor(t2/BinSize);

    t1b = unique(t1b);
    t2b = unique(t2b);

    CorrCount = ones(1,2*MaxDelay+1)*(length(t1b) + length(t2b));

    for i=1:(2*MaxDelay+1)
        t2bShifted = t2b + (i - MaxDelay - 1);
        CorrCount(i) = CorrCount(i) - length(unique([t1b(:) ; t2bShifted(:)]));
    end


    
    cc = CorrCount;
    
%     tcc = crosscorrspike(t1,t2,100);
else
    t1 = handles.SpikeTimes{CellNb};
    
    BinSize = 2;%1 ms

    MaxDelay = 100;% in BinSize

    %Here it starts

    t1b = floor(t1/BinSize);

    t1b = unique(t1b);

    CorrCount = ones(1,2*MaxDelay+1)*(2*length(t1b) );

    for i=1:(2*MaxDelay+1)
        t2bShifted = t1b + (i - MaxDelay - 1);
        CorrCount(i) = CorrCount(i) - length(unique([t1b(:) ; t2bShifted(:)]));
    end

    cc = CorrCount;

    cc(101) = 0;
end
% end

bar(handles.CrossCorrWin,(-100:100),cc);
xlabel(handles.CrossCorrWin,'Delay (ms)');
set(handles.CrossCorrWin,'xlim',[-100 100])


function ClearCrossCorr(hObject, eventdata, handles)

if isfield(handles,'CrossCorrWin')
    plot(handles.CrossCorrWin,[0 1],[0 0]);

    guidata(hObject, handles);

end



function Yspacing_Callback(hObject, eventdata, handles)
% hObject    handle to Yspacing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Yspacing as text
%        str2double(get(hObject,'String')) returns contents of Yspacing as a double

PlotData(handles)


% --- Executes during object creation, after setting all properties.
function Yspacing_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Yspacing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ForwardNavigate.
function ForwardNavigate_Callback(hObject, eventdata, handles)
% hObject    handle to ForwardNavigate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));
t = handles.SpikeTimes{CellNb}*(handles.SamplingRate/1000);

set(handles.EnableWaveforms,'Value', 1);

% duration = round(str2double(get(handles.Xscale,'String')));

duration = size(handles.templates,2);

if duration/2 == round(duration/2)
    duration = duration + 1;
end


AboveT = find(t> (handles.DataStartPt +(duration+1)/2 - 1) );

if ~isempty(AboveT)
    handles.DataStartPt =t(AboveT(1)) - (duration+1)/2 + 1;
    handles.Amp2Fit = handles.Amplitudes{CellNb}(AboveT(1));
    handles.AmpIndex = AboveT(1);
    if handles.DataStartPt<0
        handles.DataStartPt=0;
    end
end

guidata(hObject, handles);

DisplayRawData(hObject, eventdata, handles);


    
% --- Executes on button press in BackwardNavigate.
function BackwardNavigate_Callback(hObject, eventdata, handles)
% hObject    handle to BackwardNavigate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.EnableWaveforms,'Value', 1);

CellNb = str2num(get(handles.TemplateNb,'String'));
t = handles.SpikeTimes{CellNb}*(handles.SamplingRate/1000);


% duration = round(str2double(get(handles.Xscale,'String')));

duration = size(handles.templates,2);

if duration/2 == round(duration/2)
    duration = duration + 1;
end


BelowT = find(t< (handles.DataStartPt +(duration+1)/2 - 1) );
    

if ~isempty(BelowT)
    handles.DataStartPt =t(BelowT(end)) - (duration+1)/2 + 1;
    handles.Amp2Fit = handles.Amplitudes{CellNb}(BelowT(end));
    handles.AmpIndex = BelowT(end);
    if handles.DataStartPt<0
        handles.DataStartPt=0;
    end
end

guidata(hObject, handles);

DisplayRawData(hObject, eventdata, handles);


function Xscale_Callback(hObject, eventdata, handles)
% hObject    handle to Xscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Xscale as text
%        str2double(get(hObject,'String')) returns contents of Xscale as a double

PlotData(handles)

% --- Executes during object creation, after setting all properties.
function Xscale_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Xscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in SelectAmp.
function SelectAmp_Callback(hObject, eventdata, handles)
% hObject    handle to SelectAmp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


axes(handles.AmpTimeWin);
[xp,yp] = ginput(1);

CellNb = str2num(get(handles.TemplateNb,'String'));

% duration = round(str2double(get(handles.Xscale,'String')));

duration = size(handles.templates,2);

if duration/2 == round(duration/2)
    duration = duration + 1;
end

Y = handles.Amplitudes{CellNb};

X = double(handles.SpikeTimes{CellNb});

xp = xp / max(X);

X = X / max(X);

d = (X - xp).^2 + (Y - yp).^2;

[m,idp] = min(d);

handles.DataStartPt = double(handles.SpikeTimes{CellNb}(idp)*(handles.SamplingRate/1000))  - (duration+1)/2 + 1 ;

handles.Amp2Fit = handles.Amplitudes{CellNb}(idp);
handles.AmpIndex = idp;

if handles.DataStartPt<0
    handles.DataStartPt=0;
end

guidata(hObject, handles);

DisplayRawData(hObject, eventdata, handles);


function DisplayRawData(hObject, eventdata, handles)
%Extract the raw data and prepare it for display

if get(handles.EnableWaveforms,'Value')==1

    tstart = handles.DataStartPt;

    NbElec = handles.NelecTot;

    FileStart = handles.HeaderSize + 2*NbElec*tstart;%We assume that each voltage value is written on 2 bytes. Otherwise this line must be changed. 


    duration = size(handles.templates,2);

    if duration/2 == round(duration/2)
        duration = duration + 1;
    end

    FullStart  = FileStart - size(handles.templates,2)*NbElec*2;
    FullLength = (duration + 2*size(handles.templates,2))*NbElec;

    fseek(handles.DataFid,FullStart,'bof');

    data = double(fread(handles.DataFid,FullLength,handles.DataFormat));

    if strcmp(handles.DataFormat,'uint16')
        data = data - 32767;
    end

    data = data*handles.Gain;

    data = reshape(data,[NbElec (duration + 2*size(handles.templates,2))]);

    %% Filtering

    data = data(handles.ElecPermut + 1,:);

    data = handles.WhiteSpatial*data;
    for i=1:size(data,1)
        data(i,:) = conv(data(i,:),handles.WhiteTemporal,'same');
    end

    %% Reduce the data to the portion of interest - remove also the unnecessary
    %electrodes
    handles.RawData = data(:,(size(handles.templates,2)+1):(end-size(handles.templates,2)));
    guidata(hObject, handles);
else
    rmfield(handles, 'RawData');
    guidata(hObject, handles);
end

PlotData(handles)


% --- Executes on button press in EnableWaveforms.
function EnableWaveforms_Callback(hObject, eventdata, handles)
% hObject    handle to EnableWaveforms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of EnableWaveforms
PlotData(handles)

% --- Executes on button press in ResetBtn.
function ResetButtn_Callback(hObject, eventdata, handles)
% hObject    handle to ResetBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.Xmin = min(handles.Positions(:,1));
handles.Xmax = max(handles.Positions(:,1));
handles.Ymin = min(handles.Positions(:,2));
handles.Ymax = max(handles.Positions(:,2))+1;

PlotData(handles)
