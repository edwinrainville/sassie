% wave attenuation during SASSIE play 1
% 
% J. Thomson, Apr 2023

clear all, close all

datapath =  '../L1' ;

Hs = [];
time = [];
lat = [];
lon = [];
speed = [];

counter = 0;

SWIFTfiles = dir([ datapath '/SWIFT_L1/SWIFTplay1_L1/*.mat' ] );

for i = 1:length(SWIFTfiles)
    load([ datapath '/SWIFT_L1/SWIFTplay1_L1/' SWIFTfiles(i).name ] )
   Hs(counter + [1:length(SWIFT)] ) = [SWIFT.sigwaveheight];
   time(counter + [1:length(SWIFT)] ) = [SWIFT.time];
   lat(counter + [1:length(SWIFT)] ) = [SWIFT.lat];
   lon(counter + [1:length(SWIFT)] ) = [SWIFT.lon];
   speed(counter + [1:length(SWIFT)]) = [SWIFT.driftspd];
   counter = length(Hs);
end

WGfiles = dir([ datapath '/WaveGliders_L1/*.mat' ] );

for i = 1:length(WGfiles)
    load([ datapath '/WaveGliders_L1/' WGfiles(i).name ] )
   Hs(counter + [1:length(SV3)] ) = [SV3.sigwaveheight];
   time(counter + [1:length(SV3)] ) = [SV3.time];
   lat(counter + [1:length(SV3)] ) = [SV3.lat];
   lon(counter + [1:length(SV3)] ) = [SV3.lon];
   speed(counter + [1:length(SV3)]) = [NaN];
   counter = length(Hs);
end



%% trim time series for a window with stationarity 
maxtime = datenum(2022,9,10,12,0,0); %datenum(2022,9,12,0,0,0);
mintime = datenum(2022,9,9,21,0,0); %datenum(2022,9,9,21,0,0);

figure(1), clf
plot(time,Hs,'x'), hold on
datetick

timetrim = time > maxtime | time < mintime;

Hs(timetrim) = [];
lat(timetrim) = [];
lon(timetrim) = [];
time(timetrim) = [];
speed(timetrim) = [];


figure(1),
plot(time,Hs,'rx'), hold on
datetick
ylabel('Hs [m]')

%% attenuation 

iceedgelat = 72.4;
x = deg2km(lat - iceedgelat) * 1000;
H0 = mean(Hs(x<0), "omitmissing");
alpha = regress( ( log(H0) - log(Hs(x>0)) )' , x(x>0)' ) 


figure(2), clf
%plot(lat,Hs,'x')
plot([0 x(x>0)],[H0 H0*exp(-alpha*x(x>0))],'r.','linewidth',2), hold on
% plot(x,Hs,'kx','linewidth',2), hold on
scatter(x, Hs, 15, time, 'filled')
colorbar;
cbdate;
xlabel('distance to ice edge, x [m]')
ylabel('Hs [m]')

legend(['\alpha = ' num2str(alpha)])
set(gca,'fontsize',18,'fontweight','demi')
print -dpng SASSIE_play1_waveiceattenuation.png

%% Load the raw GPS positions and plot colored by time
% load([ datapath '/SWIFT12_09-12Sep2022_reprocessedIMU_RCprefilter_displacements.mat' ] );
% figure()
% for i = 1:length(SWIFT)
%     scatter(SWIFT(i).rawlon, SWIFT(i).rawlat)
%     hold on
% end
% xlim([-151.3, -149.9])
% ylim([72.5, 72.8])
% 
% % Make colormap based on time
% axh = gca(); 
% cmap = axh.ColorOrder;
% nLines = length(SWIFT)
% cmap = repmat(cmap, ceil(length(SWIFT)/size(cmap,1)), 1); 
% colormap(axh,cmap(1:nLines,:)); 
% cbh = colorbar();
% clim([1,nLines+1]) % tick number 'n' is at the bottom of the n_th color
% ylabel(cbh,'Line number')

%% Break up the Significant Wave Heights into Time bins and Recalculate regression for each
[time_sorted,sorted_inds] = sort(time);
time_reshaped = reshape(time_sorted, [81, 5]);
hs_sorted = Hs(sorted_inds);
hs_reshaped = reshape(hs_sorted, [81, 5]);
x_sorted = x(sorted_inds);
x_reshaped = reshape(x_sorted, [81, 5]);
speed_sorted = speed(sorted_inds);
speed_reshaped = reshape(speed_sorted, [81, 5]);

% Compute attenuation rate for first window
iceedgelat = 72.4;
figure(3), clf
alpha_in_windows = zeros([5,1]);
speed_in_ice = zeros([5,1]);
speed_out_ice = zeros([5,1]);
for n = 1:5
    hs_window = hs_reshaped(:,n);
    x_window = x_reshaped(:,n);
    time_window = time_reshaped(:,n);
    speed_window = speed_reshaped(:,n);
    H0 = mean(hs_window(x_window<0), "omitmissing");
    alpha = regress( ( log(H0) - log(hs_window(x_window>0)) ) , x_window(x_window>0) ) 
    alpha_in_windows(n) = alpha;
    speed_in_ice(n) = nanmean(speed_window(x_window>0));
    speed_out_ice(n) = nanmean(speed_window(x_window<0));

    % plot([0, x_window(x_window>0)'],[H0 H0*exp(-alpha*x_window(x_window>0))'],'r.','linewidth',2), hold on
    scatter(x_window, hs_window, 15, 'filled'), hold on
    % colorbar;
    % cbdate;
    xlabel('distance to ice edge, x [m]')
    ylabel('Hs [m]')
end

%% Look at each window individually
n=5;
hs_window = hs_reshaped(:,n);
x_window = x_reshaped(:,n);
speed_window = speed_reshaped(:,n);
time_window = time_reshaped(:,n);
H0 = mean(hs_window(x_window<0), "omitmissing");
alpha = regress( ( log(H0) - log(hs_window(x_window>0)) ) , x_window(x_window>0) ) 

figure()
yyaxis left 
plot([0, x_window(x_window>0)'],[H0 H0*exp(-alpha*x_window(x_window>0))'],'r.','linewidth',2), hold on
scatter(x_window, hs_window, 15, "blue", 'filled'), hold on
% colorbar;
% cbdate;
xlabel('distance to ice edge, x [m]')
ylabel('Hs [m]')

yyaxis right
scatter(x_window, speed_window, 15, "o", 'filled'), hold on
ylabel('Drift Speed [m/s]')