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
%      *See GUI Options on GUIDEs Tools menu.  "Choose GUI allows only one
%      instance to run (singleton").
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SortingGUI

% Last Modified by GUIDE v2.5 03-Dec-2015 15:37:04

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

handles.filename = varargin{2};
handles.suffix   = varargin{3};

if ~strcmp(handles.suffix, '.mat')
    result = strsplit(handles.suffix, '.mat');
    set(handles.VersionNb, 'String', result(1));
end

handles.SamplingRate = varargin{1};

hold(handles.TemplateWin,'on');

handles.H = DATA_SortingGUI;
set(handles.Yscale, 'String', '2');
set(handles.Xscale, 'String', '2');
set(handles.XYratio, 'String', '2');
set(handles.CrossCorrMaxBin,'String','2');

if size(strfind(varargin{4}, '.mat')) > 0
    b                  = load(varargin{4}, '-mat');
    handles.Positions  = double(b.Positions);
    handles.NelecTot   = b.nb_total;
    handles.ElecPermut = b.Permutation;
else
    handles.Positions  = double(h5read(varargin{4}, '/positions'))';
    handles.ElecPermut = h5read(varargin{4}, '/permutation');
    handles.NelecTot   = h5read(varargin{4}, '/nb_total');
end

handles.H.MaxdiffX  = max(handles.Positions(:,1)) - min(handles.Positions(:,1));
handles.H.MaxdiffY  = max(handles.Positions(:,2)) - min(handles.Positions(:,2));
handles.H.zoom_coef = max(handles.H.MaxdiffX,handles.H.MaxdiffY);
handles.H.lines     = cell(3,1);

if length(varargin)<=4
    handles.RPVlim = 2;
else
    handles.RPVlim = varargin{5};
end



%% Template file: could also contain AmpLim and AmpTrend
if exist([handles.filename '.templates' handles.suffix])
    template           = load([handles.filename '.templates' handles.suffix],'-mat');
    handles.templates  = template.templates;
    handles.templates_size = size(handles.templates); 
    handles.templates_size = [handles.templates_size(1) handles.templates_size(2) handles.templates_size(3)/2];
    handles.has_hdf5   = false;
    if isfield(template, 'Tagged')
        handles.Tagged = template.Tagged;
    end
    if isfield(template, 'AmpTrend')
        handles.AmpTrend = template.AmpTrend;
    end
else
    tmpfile  = [handles.filename '.templates' handles.suffix];
    tmpfile  = strrep(tmpfile, '.mat', '.hdf5');
    info     = h5info(tmpfile);
    handles.has_hdf5  = true;
    handles.is_dense  = true;
    has_tagged        = false;
    for id=1:size(info.Datasets, 1)
        if strcmp(info.Datasets(id).Name, 'temp_x')
            handles.is_dense = false;
        end
        if strcmp(info.Datasets(id).Name, 'tagged')
            has_tagged = true;
        end
    end
    if handles.is_dense
        template = h5read(tmpfile, '/templates');
        for id=1:size(info.Datasets, 1)    
            if strcmp(info.Datasets(id).Name, 'templates')
                handles.templates_size = info.Datasets(id).Dataspace.Size;
                handles.templates_size = [handles.templates_size(3) handles.templates_size(2) handles.templates_size(1)/2];
            end
        end
    else
        handles.templates_size = double(h5read(tmpfile, '/temp_shape'));
        temp_x = double(h5read(tmpfile, '/temp_x') + 1);
        temp_y = double(h5read(tmpfile, '/temp_y') + 1); 
        temp_z = double(h5read(tmpfile, '/temp_data'));
        handles.templates = sparse(temp_x, temp_y, temp_z, handles.templates_size(1)*handles.templates_size(2), handles.templates_size(3));
        handles.templates_size = [handles.templates_size(1) handles.templates_size(2) handles.templates_size(3)/2];
    end

    if has_tagged
        handles.Tagged = h5read(tmpfile, '/tagged');
    end
    has_amptrend = false;
    for id=1:size(info.Groups, 1)
        if strcmp(info.Groups(id).Name, 'amptrend')
            has_amptrend = true;
        end
    end
    if has_amptrend
        for id=1:handles.templates_size(3)
            data = h5read(tmpfile, ['/amptrend/temp_' int2str(id - 1)]);
            ndim = numel(size(data));
            data = permute(data,[ndim:-1:1]);
            handles.AmpTrend{id} = data;
        end
    end
end

handles.to_keep         = 1:handles.templates_size(3);
handles.all_actions     = cell(0);
handles.all_cells       = cell(handles.templates_size(3), 1);
for i=1:handles.templates_size(3)
    handles.all_cells{i} = [i];
end
handles.local_template  = [];
handles.local_template2 = [];

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
                ch = fread(handles.DataFid, 1, 'uint8=>char');
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

        if exist([handles.filename '.whitening.mat'])
            a = load([handles.filename '.whitening.mat'],'-mat');
            handles.WhiteSpatial  = a.spatial;
            handles.WhiteTemporal = a.temporal;
        else
            tmpfile = [handles.filename '.basis.hdf5'];
            info     = h5info(tmpfile);
            for id=1:size(info.Datasets, 1)
                if strcmp(info.Datasets(id).Name, 'spatial')
                    handles.WhiteSpatial  = h5read(tmpfile, '/spatial');
                    ndim                  = numel(size(handles.WhiteSpatial));
                    handles.WhiteSpatial  = permute(handles.WhiteSpatial,[ndim:-1:1]);
                end
                if strcmp(info.Datasets(id).Name, 'temporal')
                    handles.WhiteTemporal = h5read(tmpfile, '/temporal');
                    ndim                  = numel(size(handles.WhiteTemporal));
                    handles.WhiteTemporal = permute(handles.WhiteTemporal,[ndim:-1:1]);
                end
                if strcmp(info.Datasets(id).Name, 'thresholds')
                    handles.Thresholds    = h5read(tmpfile, '/thresholds');
                end
            end     
        end
    end
end

%% spiketimes file
if exist([handles.filename '.spiketimes' handles.suffix],'file')    
    a = load([handles.filename '.spiketimes' handles.suffix],'-mat');
    if isfield(a, 'SpikeTimes')
        for id=1:handles.templates_size(3)
            handles.SpikeTimes{id} = a.SpikeTimes{id}/(handles.SamplingRate/1000);
        end
    else
        for id=1:handles.templates_size(3)
            handles.SpikeTimes{id} = double(eval(['a.temp_' int2str(id-1)]))/(handles.SamplingRate/1000);
        end
    end
else
    tmpfile = [handles.filename '.result' handles.suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    info    = h5info(tmpfile);
    for id = 1:size(info.Groups(1).Datasets, 1)
        data = h5read(tmpfile, ['/spiketimes/temp_' int2str(id - 1)]);
        if any(data ~= 0)
            handles.SpikeTimes{id} = double(data)/(handles.SamplingRate/1000);
        else
            handles.SpikeTimes{id} = zeros(0, 1);
        end
    end
end

%% Amplitude Limits

if exist([handles.filename '.templates' handles.suffix],'file')
    b              = load([handles.filename '.templates' handles.suffix],'-mat');
    if isfield(b, 'AmpLim')
        handles.AmpLim = b.AmpLim;
    else
        b              = load([handles.filename '.limits.mat'],'-mat');
        handles.AmpLim = b.limits;
    end
elseif exist([handles.filename '.limits.mat'],'file')
    b              = load([handles.filename '.limits.mat'],'-mat');
    handles.AmpLim = b.limits;
else
    tmpfile        = [handles.filename '.templates' handles.suffix];
    tmpfile        = strrep(tmpfile, '.mat', '.hdf5');
    handles.AmpLim = h5read(tmpfile, '/limits');
    ndim           = numel(size(handles.AmpLim));
    handles.AmpLim = permute(handles.AmpLim,[ndim:-1:1]);
end

if ~isfield(handles, 'AmpTrend')
    m = 0;
    for i=1:length(handles.SpikeTimes)
        if size(handles.SpikeTimes{i}, 1) > 1
            m = max(m,max(handles.SpikeTimes{i}));   
        end
    end
    handles.AmpTrend = cell(1,handles.templates_size(3));
    for i=1:length(handles.SpikeTimes)
        handles.AmpTrend{i}([1 2],1) = [0 m];
        handles.AmpTrend{i}([1 2],2) = [1 1];
    end    
end

if ~isfield(handles, 'Tagged')
    handles.Tagged = zeros(handles.templates_size(3),1);
end


%% stimulus file: check if it exists

if exist([handles.filename '.stim'],'file')
    a = load([handles.filename '.stim'],'-mat');
    handles.StimBeg = a.rep_begin_time;
    handles.StimEnd = a.rep_end_time;
end

if isfield(handles, 'StimBeg')
    if iscell(handles.StimBeg)
        for s_i = 1:length(handles.StimBeg)
            handles.StimBeg{s_i} = handles.StimBeg{s_i}/ (handles.SamplingRate/1000);
            handles.StimEnd{s_i} = handles.StimEnd{s_i}/ (handles.SamplingRate/1000);
        end
    else
        handles.StimBeg = handles.StimBeg/ (handles.SamplingRate/1000);
        handles.StimEnd = handles.StimEnd/ (handles.SamplingRate/1000);
    end
end


%% Amplitudes file: does not have to exist

if exist([handles.filename '.amplitudes' handles.suffix],'file') 
    a = load([handles.filename '.amplitudes' handles.suffix],'-mat');
    
    if isfield(a,'Amplitudes')
        handles.Amplitudes  = a.Amplitudes;
        handles.Amplitudes2 = a.Amplitudes2;
        for id=1:handles.templates_size(3)
            if size(handles.Amplitudes{id}, 1) ~= size(handles.Amplitudes2{id}, 1)
                handles.Amplitudes2{id} = zeros(size(handles.Amplitudes{id}));
            end
        end
    else
        for id=1:handles.templates_size(3)
            if ~isempty(eval(['a.temp_' int2str(id-1)]))
                handles.Amplitudes{id}  = double(eval(['a.temp_' int2str(id-1) '(:,1)']));
                handles.Amplitudes2{id} = double(eval(['a.temp_' int2str(id-1) '(:,2)']));
            else
                handles.Amplitudes{id}  = [];
                handles.Amplitudes2{id} = [];
            end
        end
    end
else
    tmpfile = [handles.filename '.result' handles.suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    info    = h5info(tmpfile);
    for id = 1:size(info.Groups(1).Datasets, 1)
        data        = h5read(tmpfile, ['/amplitudes/temp_' int2str(id - 1)]);
        if any(data ~= 0)
            ndim = numel(size(data));
            data = permute(data,[ndim:-1:1]);
            handles.Amplitudes{id}  = data(:, 1);
            handles.Amplitudes2{id} = data(:, 2);
        else
            handles.Amplitudes{id}  = [];
            handles.Amplitudes2{id} = [];
        end
    end
end

%% Clusters file

if exist([handles.filename '.clusters' handles.suffix], 'file')    
    a = load([handles.filename '.clusters' handles.suffix],'-mat');
    if isfield(a,'clusters') %This is the save format
        handles.clusters     = a.clusters;
        handles.BestElec     = a.BestElec;
    else 
        handles.BestElec = a.electrodes + 1;
        for id=1:handles.templates_size(1)%number of electrodes
            if ~isempty(eval(['a.clusters_' int2str(id-1)]))
                features  = double(eval(['a.data_' int2str(id-1)]));
                ClusterId = double(eval(['a.clusters_' int2str(id-1)]));
                Values    = sort(unique(ClusterId),'ascend');
                if Values(1) == -1
                    Values(1) = [];
                end
                
                corresponding_template_nbs = find(a.electrodes==id-1);
                if length(corresponding_template_nbs) > 0
                    for idx=1:length(Values)
                        sf = features(find(ClusterId==Values(idx)),:);
                        handles.clusters{corresponding_template_nbs(idx)} = sf;
                    end
                end
            end
        end
    end
else
    tmpfile = [handles.filename '.clusters' handles.suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    info    = h5info(tmpfile);
    handles.BestElec = h5read(tmpfile, '/electrodes') + 1;
    if size(info.Groups, 1) > 0 %This is the save format
        for id=1:handles.templates_size(3)
            data = h5read(tmpfile, ['/clusters/temp_' int2str(id-1)]);
            ndim = numel(size(data));
            data = permute(data,[ndim:-1:1]);
            handles.clusters{id} = data;
        end
    else
        for id=1:handles.templates_size(1) %number of electrodes
            if ~isempty(h5read(tmpfile, ['/clusters_' int2str(id-1)]))
                features  = h5read(tmpfile, ['/data_' int2str(id-1)]);
                ndim      = numel(size(features));
                features  = permute(features,[ndim:-1:1]);
                ClusterId = h5read(tmpfile, ['/clusters_' int2str(id-1)]);
                ndim      = numel(size(ClusterId));
                ClusterId = permute(ClusterId,[ndim:-1:1]);
                Values    = sort(unique(ClusterId),'ascend');
                if Values(1) == -1
                    Values(1) = [];
                end
                
                corresponding_template_nbs = find(handles.BestElec == id);
                if length(corresponding_template_nbs) > 0
                    for idx=1:length(Values)
                        sf = features(find(ClusterId==Values(idx)), :);
                        handles.clusters{corresponding_template_nbs(idx)} = sf;
                    end
                end
            end
        end
    end
end


%% overlap

if exist([handles.filename '.overlap' handles.suffix], 'file')
    a               = load([handles.filename '.overlap' handles.suffix],'-mat');
    if isfield(a, 'overlap')
        handles.overlap = a.overlap/(handles.templates_size(1) * handles.templates_size(2));
    else
        handles.overlap = a.maxoverlap/(handles.templates_size(1) * handles.templates_size(2));
    end
else
    tmpfile = [handles.filename '.templates' handles.suffix];
    tmpfile = strrep(tmpfile, '.mat', '.hdf5');
    handles.overlap = h5read(tmpfile, '/maxoverlap')/(handles.templates_size(1) * handles.templates_size(2));
end

if size(handles.overlap, 1) == handles.templates_size(3)*2
    handles.overlap = handles.overlap(1:end/2,1:end/2,:);
end

handles.TemplateDisplayRatio = 0.9;


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

handles.H.zoom_coef  = handles.H.zoom_coef/1.2;
is_changes = set_TemplateWin_XY_Lims(handles);


% --- Executes on button press in ZoonOutBtn.
function ZoonOutBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ZoonOutBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


handles.H.zoom_coef  = handles.H.zoom_coef*1.2;
is_changes = set_TemplateWin_XY_Lims(handles);
if ~is_changes
    handles.H.zoom_coef  = handles.H.zoom_coef/1.2;
end


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

CellNb   = str2num(get(handles.TemplateNb,'String'));

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

ClearCrossCorr(hObject, eventdata, handles);
set(handles.SimilarNb, 'String', '1');
set(handles.TwoView, 'Value', 0);
guidata(hObject, handles);

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
if CellNb < length(handles.to_keep)
    CellNb = CellNb + 1;
end
set(handles.TemplateNb,'String',int2str(CellNb));
TemplateNb_Callback(hObject, eventdata, handles);
    

% --- Executes on button press in TemplateNbMinus.
function TemplateNbMinus_Callback(hObject, eventdata, handles)
% hObject    handle to TemplateNbMinus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


CellNb = str2num(get(handles.TemplateNb,'String'));
if CellNb > 1
    CellNb = CellNb - 1;
end
set(handles.TemplateNb,'String',int2str(CellNb));
TemplateNb_Callback(hObject, eventdata, handles);


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
if CellNb2 < size(handles.to_keep, 2)
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

if handles.Tagged(CellNb) < 5
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


% --- Executes on button press in SuggestSimilar.
function SuggestSimilar_Callback(hObject, eventdata, handles)
% hObject    handle to SuggestSimilar (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[SimilarNb, CellNb, val, IdTempl] = find_similar_templates(handles);

if IdTempl == str2double(get(handles.Template2Nb, 'String')) % if the second template is already the good one, plot the next one
    set(handles.SimilarNb, 'String', int2str(SimilarNb+1));
    SuggestSimilar_Callback(hObject, eventdata, handles);
else
    plot_similar_template(hObject, handles, CellNb, IdTempl, SimilarNb, val); 
end


% --- Executes on button press in SuggestSimilarPrev.
function SuggestSimilarPrev_Callback(hObject, eventdata, handles)

[SimilarNb, CellNb, val, IdTempl] = find_similar_templates(handles);

if IdTempl == str2double(get(handles.Template2Nb, 'String')) % if the second template is already the good one, plot the previous one
    if SimilarNb>1
        set(handles.SimilarNb, 'String', int2str(SimilarNb-1));
        SuggestSimilar_Callback(hObject, eventdata, handles);
    end
else
    plot_similar_template(hObject, handles, CellNb, IdTempl, SimilarNb, val);
end


function [SimilarNb, CellNb, val, IdTempl] = find_similar_templates(handles)

SimilarNb = str2num(get(handles.SimilarNb,'String'));
CellNb    = str2num(get(handles.TemplateNb,'String'));
comp      = squeeze(handles.overlap(CellNb,:));

if get(handles.SameElec,'Value')~=0
    comp(handles.BestElec ~= handles.BestElec(CellNb)) = 0;
end

comp     = max(comp,[],1);
[val,id] = sort(comp,'descend');
IdTempl  = id(SimilarNb+1);%The first one is the template with itself. 

if get(handles.SameElec,'Value')~=0 & val(SimilarNb+1)==0
    disp('No more templates to compare in the same electrode')
end

function plot_similar_template(hObject,handles, CellNb, IdTempl, SimilarNb, val)

set(handles.Template2Nb,'String',int2str(IdTempl))
set(handles.TwoView,'Value',1)

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

CellNb  = str2num(get(handles.TemplateNb,'String'));
CellNb2 = str2num(get(handles.Template2Nb,'String'));

t = [handles.SpikeTimes{CellNb}(:) ; handles.SpikeTimes{CellNb2}(:) ];
[t,id] = sort(t,'ascend');

a = [handles.Amplitudes{CellNb}(:) ; handles.Amplitudes{CellNb2}(:) ];
a = a(id);

b = [handles.Amplitudes2{CellNb}(:) ; handles.Amplitudes2{CellNb2}(:) ];
b = b(id);

nb_templates    = length(handles.SpikeTimes);
myslice         = [(1:CellNb2-1) (CellNb2+1:nb_templates)];
handles.to_keep = handles.to_keep(myslice);
handles.SpikeTimes{CellNb}   = t;
handles.Amplitudes{CellNb}   = a;
handles.Amplitudes2{CellNb}  = b;
handles.SpikeTimes(CellNb2)  = [];
handles.Amplitudes(CellNb2)  = [];
handles.Amplitudes2(CellNb2) = [];
handles.AmpLim(CellNb2,:)    = [];
handles.AmpTrend(CellNb2)    = [];
handles.clusters(CellNb2)    = [];
handles.BestElec(CellNb2)    = [];
handles.Tagged(CellNb2)      = [];
handles.overlap(CellNb2,:)   = [];
handles.overlap(:,CellNb2)   = [];

nb_actions = length(handles.all_actions);
handles.all_actions{nb_actions + 1} = struct('action', 'merge', 'source', CellNb, 'target', CellNb2);

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


function PlotData(handles, window_update)

CellNb   = str2num(get(handles.TemplateNb,'String'));
CellNb2  = str2num(get(handles.Template2Nb,'String'));
RCellNb  = handles.to_keep(CellNb);
RCellNb2 = handles.to_keep(CellNb2);

set(handles.Nspk1, 'String', int2str(length(handles.SpikeTimes{CellNb})));
if ~isempty(CellNb2) && (CellNb2>0) && (CellNb2 <= length(handles.SpikeTimes))
    set(handles.Nspk2, 'String', int2str(length(handles.SpikeTimes{CellNb2})));
end

set(handles.ElecNb,'String',int2str(handles.BestElec(CellNb)))
set(handles.ElecNb2,'String',int2str(handles.BestElec(CellNb2)))

ViewMode = 1 + get(handles.TwoView,'Value');

%% PLOT TEMPLATE
if handles.has_hdf5
    if handles.is_dense
        tmpfile  = [handles.filename '.templates' handles.suffix];
        tmpfile  = strrep(tmpfile, '.mat', '.hdf5');
        handles.local_template  = h5read(tmpfile, '/templates', [RCellNb 1 1], [1 handles.templates_size(2) handles.templates_size(1)]);
        handles.local_template2 = h5read(tmpfile, '/templates', [RCellNb2 1 1], [1 handles.templates_size(2) handles.templates_size(1)]);
        ndim                    = numel(size(handles.local_template));
        handles.local_template  = permute(handles.local_template,[ndim:-1:1]);
        ndim                    = numel(size(handles.local_template2));
        handles.local_template2 = permute(handles.local_template2,[ndim:-1:1]);
    else
        handles.local_template  = full(reshape(handles.templates(:, RCellNb), handles.templates_size(2), handles.templates_size(1)))';
        handles.local_template2 = full(reshape(handles.templates(:, RCellNb2), handles.templates_size(2), handles.templates_size(1)))';
    end
else
    handles.local_template  = handles.templates(:, :, RCellNb);
    handles.local_template2 = handles.templates(:, :, RCellNb2);
end 

GradeStr{1} = 'O';
GradeStr{2} = 'E';
GradeStr{3} = 'D';
GradeStr{4} = 'C';
GradeStr{5} = 'B';
GradeStr{6} = 'A';

set(handles.CellGrade,'String',GradeStr{handles.Tagged(CellNb)+1});

set(handles.CellGrade2,'String',GradeStr{handles.Tagged(CellNb2)+1});

ShowCorr_Callback('', '', handles);

raw_data_color = 0.8*[1 1 1];

% lines :    1 : template 1   ;  2 : template 2    ;    3  : raw
if ViewMode == 1
    
    if get(handles.EnableWaveforms,'Value')==0%Classical display with just the template
        
        if get(handles.NormalizeTempl,'Value')==0
            handles.H.last_neu_i_click = 1;
            PlotWaveform(handles, handles.local_template, str2double(get(handles.Yscale, 'String')),'k');
            if ishandle(handles.H.lines{2})
                delete(handles.H.lines{2});
            end
            if ishandle(handles.H.lines{3})
                delete(handles.H.lines{3});
            end
        else
            Yscale = max(abs(handles.local_template(:)));
            PlotWaveform(handles, handles.local_template, Yscale,'k');
        end
    else
        handles.H.last_neu_i_click = 1;  % ++++++++++ inversion
        PlotWaveform(handles, handles.local_template, str2double(get(handles.Yscale, 'String')),'k');
        handles.H.last_neu_i_click = 3;
        PlotWaveform(handles, handles.RawData, str2double(get(handles.Yscale, 'String')), raw_data_color);
        handles.H.last_neu_i_click = 1;  % ++++++++++ inversion
        PlotWaveform(handles, handles.local_template, str2double(get(handles.Yscale, 'String')),'k');
        if ishandle(handles.H.lines{2})
            delete(handles.H.lines{2});
        end
    end
    
else    
    if get(handles.NormalizeTempl,'Value')==0
        handles.H.last_neu_i_click = 1;
        PlotWaveform(handles, handles.local_template, str2double(get(handles.Yscale, 'String')),'b');
        handles.H.last_neu_i_click = 2;
        PlotWaveform(handles, handles.local_template2, str2double(get(handles.Yscale, 'String')),'r');
        if get(handles.EnableWaveforms,'Value')==0
            if ishandle(handles.H.lines{3})
                delete(handles.H.lines{3});
            end
        else
            handles.H.last_neu_i_click = 3;
            PlotWaveform(handles, handles.RawData, str2double(handles.Yscale.String), raw_data_color);
        end
    else
        PlotWaveform(handles,handles.local_template/max(abs(handles.local_template(:))),1)
        PlotWaveform(handles,handles.local_template2/max(abs(handles.local_template2(:))),1,'r')
    end
    
    set(handles.SimilarityTemplates,'String',num2str(max(squeeze(handles.overlap(CellNb,CellNb2)))))
end

if nargin == 1
    %% TEMPLATE COUNT

    set(handles.TemplateCountTxt,'String',[int2str(length(find(handles.Tagged>0))) '/' int2str(size(handles.to_keep, 2))])
    
    %% PLOT CLUSTER
    FeatureX = str2num(get(handles.FeatureX,'String'));
    FeatureY = str2num(get(handles.FeatureY,'String'));

    plot(handles.ClusterWin,handles.clusters{CellNb}(:,FeatureX),handles.clusters{CellNb}(:,FeatureY),'.')

    if (ViewMode==2) & (handles.BestElec(CellNb)==handles.BestElec(CellNb2))
        hold(handles.ClusterWin,'on')
        plot(handles.ClusterWin,handles.clusters{CellNb2}(:,FeatureX),handles.clusters{CellNb2}(:,FeatureY),'r.')
        hold(handles.ClusterWin,'off')
    end

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
    NbRPV = sum(ISI<=handles.RPVlim);
    RatioRPV = sum(ISI<=handles.RPVlim)/lenISI;
    if get(handles.BigISI,'Value')==0
        ISI = ISI(ISI<=25);
        hist(handles.ISIwin,ISI,100);
    else
        ISI = ISI(ISI<=200);
        hist(handles.ISIwin,ISI,200);
    end
    XL = get(handles.ISIwin, 'XLim');
    set(handles.ISIwin, 'XLim', [0 XL(2)]);
    % x = (0:2:26);
    set(handles.RPV,'String',['RPV: ' num2str(RatioRPV*100) '%, ' int2str(NbRPV) '/' int2str(lenISI) ]);

    %% PLOT RASTER

    if isfield(handles,'StimBeg')

        cc = hsv(length(handles.StimBeg));

        for k=1:length(handles.StimBeg)
           ColorRaster{k} = cc(k,:);
        end

        hold(handles.RasterWin,'off');

        a = plot(handles.RasterWin,0,0);
        delete(a);
        hold(handles.RasterWin,'on');
    
        line_count_init = 0;
        if ViewMode == 1 % 1 template
            [~, sep_line_l, MaxTimes] = plot_Raster( handles, CellNb, line_count_init, ColorRaster);
        else % 2 templates
            [line_count_init, sep_line_l1, MaxTimes1] = plot_Raster( handles, CellNb, line_count_init, ColorRaster);
            template_sep_line = sep_line_l1(end);
            [~, sep_line_l2, MaxTimes2] = plot_Raster( handles, CellNb2, line_count_init, ColorRaster);
            MaxTimes = max(MaxTimes1, MaxTimes2);
            sep_line_l = union(sep_line_l1, sep_line_l2);

            plot(handles.RasterWin,[0 MaxTimes],[template_sep_line template_sep_line],'k','LineWidth',2);
        end
        if MaxTimes > 0
            set(handles.RasterWin,'XLim',[0, MaxTimes]);
        end
        for LineCount=sep_line_l(:)'
            plot(handles.RasterWin,[0 MaxTimes],[LineCount LineCount],'k');
        end
        set(handles.RasterWin,'YLim',[0 sep_line_l(end)]);
    end
end


function [fr_times, fr_repeat, line_count_out] = find_spikes_for_raster(rep_begin_time, rep_end_time, spk_time, line_count_init)

LineCount = line_count_init;
fr_times = [];
fr_repeat = [];
for i=1:length(rep_begin_time)
    times = spk_time(spk_time >= rep_begin_time(i) & spk_time < rep_end_time(i)) ...
        - rep_begin_time(i);
    fr_times = [fr_times ; times(:)];
    fr_repeat = [fr_repeat ; LineCount*ones(length(times),1)];
    LineCount = LineCount + 1;
end
line_count_out = LineCount;


function [fr_times, fr_repeat, line_count_out] = find_spikes_for_raster2(rep_begin_time, rep_end_time, spk_time, line_count_init)

spk_time  = spk_time(:);
rep_times = [rep_begin_time(:)'; rep_end_time(:)'];
rep_times = rep_times(:);

[~, fr_repeat] = histc( spk_time, rep_times);
keep = mod(fr_repeat,2) == 1;

fr_times = spk_time(keep(:)) - rep_times(fr_repeat(keep(:)));

fr_repeat = (fr_repeat(keep)-1)/2 + line_count_init;

line_count_out = line_count_init + length(rep_begin_time);


function [line_count_out, sep_line_l, MaxTimes] = plot_Raster( handles, CellNb, line_count_init, ColorRaster)

LineCount  = line_count_init;
sep_line_l = LineCount;
LineCount  = LineCount + 1;
sep_line_l = [];
MaxTimes   = 0;
for k=1:length(handles.StimBeg)
    
    [fr_times, fr_repeat, LineCount] = find_spikes_for_raster2(handles.StimBeg{k}, handles.StimEnd{k}, handles.SpikeTimes{CellNb}, LineCount);

    sep_line_l(end+1) = LineCount;
    
    if ~isempty(fr_times)
        MaxTimes = max(MaxTimes,max(fr_times));
    end

    plot(handles.RasterWin,fr_times,fr_repeat,'.','color',ColorRaster{k});
end
line_count_out = LineCount+1;


function PlotWaveform(handles,waveform,Yscale,wcolor,Width)
%Plot the waveform matrix.
%tstart and duration might be unused. 

if nargin<3
    Yscale = str2double(get(handles.Yscale,'String'));
end

if nargin<4
    wcolor = [0 0 1];
end

if nargin<5
    Width=1;
end

Xspacing = str2double(get(handles.Xscale, 'String'));

Coor = handles.Positions;

if ishandle(handles.H.lines{handles.H.last_neu_i_click})
    delete(handles.H.lines{handles.H.last_neu_i_click});
end

if handles.H.last_neu_i_click == 1 % it is the principal electrode
    m = max(abs(waveform),[],2);
    [~,elecM] = max(m);
    handles.H.elecMx = Coor(elecM,1);
    handles.H.elecMy = Coor(elecM,2);
    
    max_t = max(waveform(:));
    min_t = min(waveform(:));
    
    handles.H.fullX   = [min(handles.Positions(:,1)) max(handles.Positions(:,1))];
    handles.H.fullY   = [min(handles.Positions(:,2)) max(handles.Positions(:,2))];

    handles.H.marginX = [ - Xspacing/1.5 , Xspacing/1.5 ];
    handles.H.marginY = [min_t - Yscale/2, max_t + Yscale/2 ];
end

[X,Y] = deal(zeros(size(waveform,2),handles.templates_size(1)));
for i = 1:handles.templates_size(1) % idElec=1:size(handles.templates,1);
    elecX = Coor(i,1);
    elecY = Coor(i,2);
    X(:,i) = (elecX)+ linspace(-Xspacing/2, Xspacing/2, size(waveform,2));
    Y(:,i) = waveform(i,:)/Yscale + (elecY);
end

X = X*str2double(get(handles.XYratio, 'String'));
handles.H.lines{handles.H.last_neu_i_click} = plot(handles.TemplateWin,X,Y,'color',wcolor,'LineWidth',Width);
set_TemplateWin_XY_Lims(handles);
set(handles.TemplateWin,'Xtick',[],'Ytick',[]);


function is_changes = set_TemplateWin_XY_Lims(handles)

xlim_old = get(handles.TemplateWin, 'XLim');
ylim_old = get(handles.TemplateWin, 'YLim');
x_surround = handles.H.elecMx*str2double(get(handles.XYratio, 'String')) + handles.H.zoom_coef*[-1, 1];

x_limits =  (handles.H.fullX+ handles.H.marginX)*str2double(get(handles.XYratio, 'String'));
x_surround = max(x_surround, x_limits(1));
x_surround = min(x_surround, x_limits(2));
set(handles.TemplateWin, 'XLim', x_surround);
y_surround = handles.H.elecMy + handles.H.zoom_coef*[-1, 1];

y_limits =  handles.H.fullY + handles.H.marginY;
y_surround = max(y_surround, y_limits(1));
y_surround = min(y_surround, y_limits(2));
set(handles.TemplateWin, 'YLim', y_surround);

is_changes = (~isequal(xlim_old, x_surround)) || (~isequal(ylim_old, y_surround));




% --- Executes on button press in KillBtn.
function KillBtn_Callback(hObject, eventdata, handles)
% hObject    handle to KillBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CellNb = str2num(get(handles.TemplateNb,'String'));

nb_templates                = length(handles.SpikeTimes);
myslice                     = [(1:CellNb-1) (CellNb+1:nb_templates)];
handles.to_keep             = handles.to_keep(myslice);
handles.SpikeTimes(CellNb)  = [];
handles.Amplitudes(CellNb)  = [];
handles.Amplitudes2(CellNb) = [];
handles.AmpLim(CellNb,:)    = [];
handles.AmpTrend(CellNb)    = [];
handles.clusters(CellNb)    = [];
handles.BestElec(CellNb)    = [];
handles.Tagged(CellNb)      = [];
handles.overlap(CellNb,:)   = [];
handles.overlap(:,CellNb)   = [];

nb_actions = length(handles.all_actions);
handles.all_actions{nb_actions + 1} = struct('action', 'remove', 'source', CellNb);

guidata(hObject, handles);

if CellNb == nb_templates
    TemplateNbMinus_Callback(hObject, eventdata, handles);
else
    PlotData(handles)
end


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

set(handles.SaveBtn, 'String', 'In progress');
guidata(hObject, handles);

%% Template file: could also contain AmpLim and AmpTrend

suffix  = get(handles.VersionNb, 'String');
if iscell(suffix)
    suffix  = suffix{1};
end
overlap = handles.overlap * (handles.templates_size(1) * handles.templates_size(2));

output_file_temp = [handles.filename '.templates' suffix '.hdf5'];
nb_templates     = size(handles.to_keep, 2);
tmp_templates    = [handles.filename '.templates-tmp' suffix '.hdf5'];
tmpfile          = [handles.filename '.templates' handles.suffix];

if exist(tmp_templates,'file')
    delete(tmp_templates);
end

mysize = int32([handles.templates_size(1) handles.templates_size(2) 2*nb_templates])
h5create(tmp_templates, '/temp_shape', size(mysize))
h5write(tmp_templates, '/temp_shape', mysize)
new_templates = sparse(handles.templates_size(1)*handles.templates_size(2), 2*nb_templates);
for count=1:nb_templates
    new_templates(:, count)                = handles.templates(:, handles.to_keep(count));
    new_templates(:, count + nb_templates) = handles.templates(:, handles.to_keep(count) + handles.templates_size(3));
end

[x, y, z] = find(new_templates);
h5create(tmp_templates, '/temp_x', size(x));
h5create(tmp_templates, '/temp_y', size(y));
h5create(tmp_templates, '/temp_data', size(z));
h5write(tmp_templates, '/temp_x', int32(x) - 1);
h5write(tmp_templates, '/temp_y', int32(y) - 1);
h5write(tmp_templates, '/temp_data', single(z));




%h5create(tmp_templates, '/templates', [2*nb_templates handles.templates_size(2) handles.templates_size(1)])
%nb_to_write = 100;
%tmp_count   = 1;
%differences = [diff(handles.to_keep) 0];
%while tmp_count <= nb_templates
%    contiguous  = find(differences(tmp_count:nb_templates) ~= 1);
%    if isempty(contiguous)
%        local_write = min(nb_to_write, nb_templates - tmp_count);
%    else
%        local_write = min(nb_to_write, contiguous(1));
%    end
%    temp_1 = handles.to_keep(tmp_count:tmp_count + local_write - 1);
%    temp_2 = handles.to_keep(tmp_count:tmp_count + local_write - 1) + handles.templates_size(3);
%    if handles.has_hdf5
%        tmpfile    = [handles.filename '.templates' handles.suffix];
%        tmpfile    = strrep(tmpfile, '.mat', '.hdf5');
%        to_write_1 = h5read(tmpfile, '/templates', [temp_1(1) 1 1], [local_write handles.templates_size(2) handles.templates_size(1)]);
%        to_write_2 = h5read(tmpfile, '/templates', [temp_2(1) 1 1], [local_write handles.templates_size(2) handles.templates_size(1)]);
%    else
%        to_write_1 = handles.templates(:, :, temp_1);
%        to_write_2 = handles.templates(:, :, temp_2);
%        ndim       = numel(size(to_write_1));
%        to_write_1 = permute(to_write_1,[ndim:-1:1]);
%        to_write_2 = permute(to_write_2,[ndim:-1:1]);
%    end
%    h5write(tmp_templates, '/templates', to_write_1, [tmp_count 1 1], [local_write handles.templates_size(2) handles.templates_size(1)]);
%    h5write(tmp_templates, '/templates', to_write_2, [tmp_count+nb_templates 1 1], [local_write handles.templates_size(2) handles.templates_size(1)]); 
%    tmp_count = tmp_count + local_write;
%end

delete(output_file_temp);
movefile(tmp_templates, output_file_temp)

h5create(output_file_temp, '/limits', size(transpose(handles.AmpLim)));
h5write(output_file_temp, '/limits', transpose(handles.AmpLim));
h5create(output_file_temp, '/maxoverlap', size(transpose(overlap)));
h5write(output_file_temp, '/maxoverlap', transpose(overlap));
h5create(output_file_temp, '/tagged', size(transpose(handles.Tagged)));
h5write(output_file_temp, '/tagged', transpose(handles.Tagged));
for id=1:nb_templates
    key = ['/amptrend/temp_' int2str(id - 1)];
    h5create(output_file_temp, key, size(transpose(handles.AmpTrend{id})));
    h5write(output_file_temp, key, transpose(handles.AmpTrend{id}));
end

output_file = [handles.filename '.result' suffix '.hdf5'];
delete(output_file);
for id=1:nb_templates
    key = ['/spiketimes/temp_' int2str(id - 1)];
    if size(handles.SpikeTimes{id}, 1) == 0
        to_write = zeros(1);        
    else
        to_write = transpose(handles.SpikeTimes{id}*(handles.SamplingRate/1000));
    end
    h5create(output_file, key, size(to_write));
    h5write(output_file, key, to_write);
    key = ['/amplitudes/temp_' int2str(id - 1)];
    if size(handles.Amplitudes{id}, 1) == 0
        to_write = zeros(1);        
    else
        to_write = transpose([handles.Amplitudes{id} handles.Amplitudes2{id}]);
    end
    h5create(output_file, key, size(to_write));
    h5write(output_file, key, to_write);
end


%% Clusters file
output_file = [handles.filename '.clusters' suffix '.hdf5'];
delete(output_file);
h5create(output_file, '/electrodes', size(transpose(handles.BestElec)));
h5write(output_file, '/electrodes', transpose(handles.BestElec));
for id=1:nb_templates
    key = ['/clusters/temp_' int2str(id - 1)];
    to_write = transpose(handles.clusters{id});
    h5create(output_file, key, size(to_write));
    h5write(output_file, key, to_write);
end

%%
if handles.has_hdf5 && strcmp(output_file_temp, tmpfile)
    handles.to_keep = 1:nb_templates;
    handles.templates_size(3) = nb_templates;
end
guidata(hObject, handles);

set(handles.SaveBtn, 'String', 'Save');


% --- Executes on button press in SplitBtn.
function SplitBtn_Callback(hObject, eventdata, handles)
% hObject    handle to SplitBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%Duplicate the template

CellNb              = str2num(get(handles.TemplateNb,'String'));
nb_templates        = length(handles.SpikeTimes);
myslice             = [(1:CellNb) (CellNb:nb_templates)];
handles.SpikeTimes  = handles.SpikeTimes(myslice);
handles.Amplitudes  = handles.Amplitudes(myslice);
handles.Amplitudes2 = handles.Amplitudes2(myslice);
handles.Tagged      = handles.Tagged(myslice);
handles.AmpLim      = handles.AmpLim(myslice,:);
handles.AmpTrend    = handles.AmpTrend(myslice);
handles.clusters    = handles.clusters(myslice);
handles.BestElec    = handles.BestElec(myslice);
handles.overlap     = handles.overlap(myslice,:);
handles.overlap     = handles.overlap(:,myslice);
handles.to_keep     = handles.to_keep(myslice);

nb_actions = length(handles.all_actions);
handles.all_actions{nb_actions + 1} = struct('action', 'split', 'source', CellNb);

%Remove the amplitudes/spiketimes in or out of the amp lims

Trend  = interp1(handles.AmpTrend{CellNb}(:,1),handles.AmpTrend{CellNb}(:,2),handles.SpikeTimes{CellNb}(:));
ToKeep = Trend*handles.AmpLim(CellNb,1) <= handles.Amplitudes{CellNb} & Trend*handles.AmpLim(CellNb,2) >= handles.Amplitudes{CellNb};

handles.SpikeTimes{CellNb}    = handles.SpikeTimes{CellNb}(find(ToKeep));
handles.Amplitudes{CellNb}    = handles.Amplitudes{CellNb}(find(ToKeep));
handles.Amplitudes2{CellNb}   = handles.Amplitudes2{CellNb}(find(ToKeep));
handles.SpikeTimes{CellNb+1}  = handles.SpikeTimes{CellNb+1}(find(~ToKeep));
handles.Amplitudes{CellNb+1}  = handles.Amplitudes{CellNb+1}(find(~ToKeep));
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

ViewMode = 1 + get(handles.TwoView,'Value');
BinSize  = str2double(get(handles.CrossCorrMaxBin,'String'));
MaxDelay = 50; % in BinSize

%% PLOT CROSS-CORR
% if 0%INSTALL GCC AND COMPILE crosscorrspike.cpp
if ViewMode==2
    t1 = handles.SpikeTimes{CellNb};
    t2 = handles.SpikeTimes{CellNb2};
    
    %Here it starts

    t1b = round(t1/BinSize);
    t2b = round(t2/BinSize);
    t1b = unique(t1b);
    t2b = unique(t2b);

    CorrCount = zeros(1,2*MaxDelay+1);
    for i=1:(2*MaxDelay+1)
        CorrCount(i) =  sum(ismember(t1b(:), t2b(:) + (i - MaxDelay - 1)));
    end
    
    cc = CorrCount;
    
    set(handles.CrossWinStyle, 'String', 'Cross-Corr');
else
    t1 = handles.SpikeTimes{CellNb};

    %Here it starts
    t1b = round(t1/BinSize);
    t1b = unique(t1b);
    CorrCount = zeros(1,2*MaxDelay+1);
    for i=1:(2*MaxDelay+1)
        if i ~= (MaxDelay+1)
            CorrCount(i) =  sum(ismember(t1b(:), t1b(:) + (i - MaxDelay - 1)));
        end
    end
    
    cc = CorrCount;
    %     cc(MaxDelay+1) = 0;
    
    set(handles.CrossWinStyle, 'String', 'Auto-Corr');
end

bar(handles.CrossCorrWin,(-MaxDelay:MaxDelay)*BinSize,cc);
set(handles.CrossCorrWin,'xlim',[-MaxDelay,MaxDelay]*BinSize)

function ClearCrossCorr(hObject, eventdata, handles)

if isfield(handles,'CrossCorrWin')
    plot(handles.CrossCorrWin,[0 1],[0 0]);

    guidata(hObject, handles);

end



function Yscale_Callback(hObject, eventdata, handles)
% hObject    handle to Yspacing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Yspacing as text
%        str2double(get(hObject,'String')) returns contents of Yspacing as a double

PlotData(handles, 1)


% --- Executes during object creation, after setting all properties.
function Yscale_CreateFcn(hObject, eventdata, handles)
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
duration = handles.templates_size(2);

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

duration = handles.templates_size(2);

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

PlotData(handles, 1)

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

duration = handles.templates_size(2);

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


    duration = handles.templates_size(2);

    if duration/2 == round(duration/2)
        duration = duration + 1;
    end

    FullStart  = FileStart - handles.templates_size(2)*NbElec*2;
    FullLength = (duration + 2*handles.templates_size(2))*NbElec;

    fseek(handles.DataFid,FullStart,'bof');

    data = double(fread(handles.DataFid,FullLength,handles.DataFormat));

    if strcmp(handles.DataFormat,'uint16')
        data = data - 32767;
    end

    data = data*handles.Gain;

    data = reshape(data,[NbElec (duration + 2*handles.templates_size(2))]);

    %% Filtering

    data = data(handles.ElecPermut + 1,:);

    if isfield(handles, 'WhiteSpatial')
        data = handles.WhiteSpatial*data;
    end
    if isfield(handles, 'WhiteTemporal')
        for i=1:size(data,1)
            data(i,:) = conv(data(i,:),handles.WhiteTemporal,'same');
        end
    end

    %% Reduce the data to the portion of interest - remove also the unnecessary
    %electrodes
    handles.RawData = data(:,(handles.templates_size(2)+1):(end-handles.templates_size(2)));
    guidata(hObject, handles);
else
    rmfield(handles, 'RawData');
    guidata(hObject, handles);
end

PlotData(handles, 1)


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

handles.H.zoom_coef = max(handles.H.MaxdiffX*str2double(get(handles.XYratio, 'String')), handles.H.MaxdiffY);
ZoonOutBtn_Callback(hObject, eventdata, handles);
PlotData(handles, 1)

% --- Executes on button press in KillEs.
function KillEs_Callback(hObject, eventdata, handles)
% hObject    handle to KillEs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

choice = questdlg('Are you sure?', 'Kill All E', 'Yes', 'No way','No way');
% Handle response
switch choice
    case 'Yes'
        wdw = find(handles.Tagged==1); % 1 corresponds to E

        if any(wdw)
            handles.to_keep(wdw)     = [];
            handles.SpikeTimes(wdw)  = [];
            handles.Amplitudes(wdw)  = [];
            handles.Amplitudes2(wdw) = [];
            handles.AmpLim(wdw,:)    = [];
            handles.AmpTrend(wdw)    = [];
            handles.clusters(wdw)    = [];
            handles.BestElec(wdw)    = [];
            handles.Tagged(wdw)      = [];
            handles.overlap(wdw,:)   = [];
            handles.overlap(:,wdw)   = [];       
            guidata(hObject, handles);
            PlotData(handles)
        end
end




% --- Executes during object creation, after setting all properties.
function SortingGUI_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SortingGUI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in KillAllEmpty.
function KillAllEmpty_Callback(hObject, eventdata, handles)
% hObject    handle to KillAllEmpty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


wdw = (cellfun('length',handles.SpikeTimes)==0);

if any(wdw)
    handles.to_keep(wdw)     = [];
    handles.SpikeTimes(wdw)  = [];
    handles.Amplitudes(wdw)  = [];
    handles.Amplitudes2(wdw) = [];
    handles.AmpLim(wdw,:)    = [];
    handles.AmpTrend(wdw)    = [];
    handles.clusters(wdw)    = [];
    handles.BestElec(wdw)    = [];
    handles.Tagged(wdw)      = [];
    handles.overlap(wdw,:)   = [];
    handles.overlap(:,wdw)   = [];
    guidata(hObject, handles);
    PlotData(handles)
end


% --- Executes on selection change in CrossCorrMaxBin.
function CrossCorrMaxBin_Callback(hObject, eventdata, handles)
% hObject    handle to CrossCorrMaxBin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns CrossCorrMaxBin contents as cell array
%        contents{get(hObject,'Value')} returns selected item from CrossCorrMaxBin
ShowCorr_Callback(hObject, eventdata, handles);

% --- Executes during object creation, after setting all properties.
function CrossCorrMaxBin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CrossCorrMaxBin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function XYratio_Callback(hObject, eventdata, handles)
% hObject    handle to XYratio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of XYratio as text
%        str2double(get(hObject,'String')) returns contents of XYratio as a double
PlotData(handles,1);

% --- Executes during object creation, after setting all properties.
function XYratio_CreateFcn(hObject, eventdata, handles)
% hObject    handle to XYratio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in TwoView.
function TwoView_Callback(hObject, eventdata, handles)
% hObject    handle to TwoView (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of TwoView

PlotData(handles)


% --- Executes on button press in switch_templates.
function switch_templates_Callback(hObject, eventdata, handles)
% hObject    handle to switch_templates (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tempString1 = get(handles.TemplateNb,'String');
set(handles.TemplateNb,'String', get(handles.Template2Nb,'String'));
set(handles.Template2Nb,'String', tempString1);

TemplateNb_Callback(hObject, eventdata, handles);
Template2Nb_Callback(hObject, eventdata, handles);
TwoView_Callback(hObject, eventdata, handles);
