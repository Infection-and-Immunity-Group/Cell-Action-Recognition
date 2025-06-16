close all;
clear all;
clc;

%% DEPENDENCIES
addpath './utils';
addpath './libs/colorbrewer/';
sigm = @(x, m) 1-(1./(1+exp(-(x-m))));

%% SETTINGS
K = 13;    % Number of time points for tracklets
dt = 15;   % Time interval between two adjacent frames
TH_AC = 2; % Cells moving below TH_AC um/min are considered arrested
ACTIONS = {'Arrested', 'Patrolling', 'Directed', 'Flowing'};
GATE_SPEED_MIN = [0, 4, 6, 10];         % um/min
GATE_SPEED_MAX = [2, 10, 25, 300];       % um/min
GATE_DIR_MIN = [0, 0.2, 0.5, 0.8];      % 0..1
GATE_DIR_MAX = [0.2, 0.5, 0.8, 1.1];    % 0..1 values greater than 1 are set to 1
GATE_AC_MIN = [0.7, 0.1, 0, 0];         % 0..1
GATE_AC_MAX = [1.1, 0.8, 0.2, 0.2];     % 0..1 values greater than 1 are set to 1
GATE_DISPL_MIN = [0, 4, 8, 20]*K/17;         % um
GATE_DISPL_MAX = [15, 20, 80, 300]*K/17;     % um
colors={'k','r','g','b'}; % Colors for different actions

%% Select single file
[fn, fn_path] = uigetfile('*.xls*', 'Select Excel File (with statistics from Imaris) to analyze');

if isequal(fn, 0)
    error('No file selected. Exiting script.');
end

%% Uncomment to visualize gates
figure;
for gg = 1:size(ACTIONS, 2)
    y0 = GATE_SPEED_MIN(gg);
    x0 = GATE_DISPL_MIN(gg);
    dy = GATE_SPEED_MAX(gg)-GATE_SPEED_MIN(gg);
    dx = GATE_DISPL_MAX(gg)-GATE_DISPL_MIN(gg);
    hold on;
    rectangle('Position',[x0,y0,dx,dy],'EdgeColor', colors{gg});
end
axis([0 60 0 30]);
xlabel('Displacement [um]');
ylabel('Speed [um/min]');
axis square;

figure;
for gg = 1:size(ACTIONS, 2)
    y0 = GATE_DIR_MIN(gg);
    x0 = GATE_AC_MIN(gg);
    dy = GATE_DIR_MAX(gg)-GATE_DIR_MIN(gg);
    dx = GATE_AC_MAX(gg)-GATE_AC_MIN(gg);
    hold on;
    rectangle('Position',[x0,y0,dx,dy],'EdgeColor', colors{gg});
end
xlabel('Arrest coefficient [0..1]');
ylabel('Confinement ratio [0..1]');
axis square;

%% Processing
curr_t = 0;

%% 1. Read all the positions
xlsNeu = xlsread(fullfile(fn_path, fn), 'Position');
xlsTime = xlsread(fullfile(fn_path, fn), 'Time');
min_time = min(xlsTime(:,4));
id1 = find(xlsTime(:,4) == min_time);
id2 = find(xlsTime(:,4) == (min_time + 1));
if(isempty(id1) || isempty(id2))
    prompt = {['Enter the time-step of the video ', fn, ' (in seconds)']};
    dlgtitle = 'Time-step';
    definput = {'30'};
    answer = inputdlg(prompt,dlgtitle,[1 100],definput);
    dt = str2num(answer{1});
else
    t1 = xlsTime(id1(1),1);
    t2 = xlsTime(id2(1),1);
    dt = t2 - t1;
end

xlsNeu(:,8) = xlsNeu(:,8);
tids = unique(xlsNeu(:,8));
T = max(xlsNeu(:,7));

%% 2. Computes measures
total_measures = zeros(9999, 7);%id, time, length, displacement, speed, dir, AC
    
count_measures = 0;
for ii = 1:numel(tids) % for each track
    curr_track_id = tids(ii);
    idx = find((xlsNeu(:,8) == curr_track_id));
    xyt = xlsNeu(idx, [1,2,7]);
    track_length = size(xyt, 1);

    if(track_length > K)  
        for jj = ceil(K/2):track_length-floor(K/2) % for each tracklet of size K
            count_measures = count_measures + 1;
            curr_xy = xyt((jj - floor(K/2)) : (jj+floor(K/2)), [1,2]);
            L = sum(sqrt(sum(diff(curr_xy).^2, 2)));
            D = sqrt(sum((curr_xy(1,:) - curr_xy(end, :)).^2));
            S = (sqrt(sum(diff(curr_xy).^2, 2))/dt)*60;
            AC0 = sum(sigm(zeros(numel(S), 1), TH_AC)) ./ numel(S);
            AC = sum(sigm(S, TH_AC)) ./ (numel(S) * AC0); %Continuous version of AC

            total_measures(count_measures, 1) = curr_track_id;
            total_measures(count_measures, 2) = xyt(jj, 3);
            total_measures(count_measures, 3) = L;
            total_measures(count_measures, 4) = D;
            total_measures(count_measures, 5) = L*60/(K*dt);
            total_measures(count_measures, 6) = D/L;
            total_measures(count_measures, 7) = AC;
        end
    end
end

total_measures = total_measures(1:count_measures, :);
mean_dir = zeros(1, T);
std_dir = zeros(1, T);

for tt = 1:T
    idx = find(total_measures(:, 2) == tt);
    mean_dir(tt) = mean(total_measures(idx, 6));
    std_dir(tt) = std(total_measures(idx, 6));
end

mean_speed = zeros(1, T);
std_speed = zeros(1, T);

for tt = 1:T
    idx = find(total_measures(:, 2) == tt);
    mean_speed(tt) = mean(total_measures(idx, 5));
    std_speed(tt) = std(total_measures(idx, 5));
end
	
%% Visualization of current file
figure; % Time series of instantaneous speed
errorbar(dt*(1:T), mean_speed, std_speed);
title('Time-series of instantaneous speed');
xlabel('Time [s]');
ylabel('Speed [um/min]');
    
%% 3. Detects classes of cells
ptsC_actions = zeros(count_measures, 1);
action_results = zeros(count_measures, 7); % [track_id, time, action, speed, displacement, directionality, arrest_coefficient]
action_labels = cell(count_measures, 1);
count_actions = zeros(1, 4);

for ii = 1:count_measures
    curr_displ = total_measures(ii, 4);
    curr_speed = total_measures(ii, 5);
    curr_dir = total_measures(ii, 6);
    curr_ac = total_measures(ii, 7);
    
    curr_gate_speed_min = curr_speed >= GATE_SPEED_MIN;
    curr_gate_dir_min = curr_dir >= GATE_DIR_MIN;
    curr_gate_ac_min = curr_ac >= GATE_AC_MIN;
    curr_gate_displ_min = curr_displ >= GATE_DISPL_MIN;
    
    curr_gate_speed_max = curr_speed <= GATE_SPEED_MAX;
    curr_gate_dir_max = curr_dir <= GATE_DIR_MAX;
    curr_gate_ac_max = curr_ac <= GATE_AC_MAX;
    curr_gate_displ_max = curr_displ <= GATE_DISPL_MAX;
    
    CURR_ACTION = (curr_gate_speed_min & curr_gate_dir_min & curr_gate_ac_min & curr_gate_displ_min);
    CURR_ACTION = CURR_ACTION & (curr_gate_speed_max & curr_gate_dir_max & curr_gate_ac_max & curr_gate_displ_max);
    
    idx = find(CURR_ACTION == 1);
    if(isempty(idx))
        ptsC_actions(ii) = 0;
        action_labels{ii} = 'Unclassified';
    else
        idx = idx(1);
        ptsC_actions(ii) = idx;
        action_labels{ii} = ACTIONS{idx};
        count_actions(idx) = count_actions(idx) + 1;
    end
    
    % Store results for export
    action_results(ii, :) = [total_measures(ii, 1), total_measures(ii, 2), ptsC_actions(ii), ...
                            curr_speed, curr_displ, curr_dir, curr_ac];
end

% Export results to Excel
[~, filename, ~] = fileparts(fn);
results_filename = [filename '_classified_actions.xlsx'];

% Create headers for the Excel file
headers = {'Track_ID', 'Time_Point', 'Action_Index', 'Action_Label', 'Speed_um_min', ...
           'Displacement_um', 'Directionality', 'Arrest_Coefficient'};

% Combine results with action labels
results_table = table(action_results(:,1), action_results(:,2), action_results(:,3), ...
                      action_labels, action_results(:,4), action_results(:,5), ...
                      action_results(:,6), action_results(:,7), ...
                      'VariableNames', headers);

% Write to Excel
writetable(results_table, results_filename);

%% Visualize percentages of cells performing actions
figure;
count_actions = count_actions ./ sum(count_actions);
bar(count_actions*100);
xticklabels(ACTIONS);
title(fn, 'Interpreter','none');
ylim([0 100]);