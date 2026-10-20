%% gain_demo.m - contraction -> FHR impulse response and gain, on a real CTU-CHB record
% Run one section at a time (Cmd+Enter). Record 1418 (cord pH 6.98) has a valid
% contraction response. Needs the Signal Processing Toolbox.

%% 1. Load record 1418 (FHR and toco, 4 Hz) and clean signal gaps
db  = fullfile(fileparts(mfilename('fullpath')), 'ctu-chb-intrapartum-cardiotocography-database-1.0.0');
fid = fopen(fullfile(db, '1418.dat'), 'r');
raw = fread(fid, [2 Inf], 'int16')' / 100;   fclose(fid);
fhr4 = raw(:, 1);   toco4 = raw(:, 2);
fhr4(fhr4 < 50 | fhr4 > 210) = NaN;          % 0 = signal lost
toco4(toco4 <= 0) = NaN;
gapsF = isnan(fhr4);                         % remember where FHR was lost
fhr4  = fillmissing(fhr4, 'linear');
toco4 = fillmissing(toco4, 'linear');

%% 2. Downsample to 0.5 Hz (one sample every 2 s) and keep the last 74 minutes
fs  = 0.5;
fhr  = decimate(decimate(fhr4, 2), 4);
toco = decimate(decimate(toco4, 2), 4);
N = 74 * 60 * fs;
fhr = fhr(end-N+1:end);   toco = toco(end-N+1:end);
t = ((0:N-1)' - N) / fs / 60;                % minutes before birth
figure(1); clf;
subplot(2,1,1); plot(t, toco); ylabel('toco'); title('Record 1418');
subplot(2,1,2); plot(t, fhr);  ylabel('FHR (bpm)'); xlabel('minutes before birth');

%% 3. Band-pass both to 0.0045-0.03 Hz (cycles of ~33 s to ~3.7 min)
% Keeps each contraction and the heart-rate change it causes; removes the slow
% baseline and the fast beat-to-beat variability.
[b1, a1] = butter(4, 0.03   / (fs/2));
[b2, a2] = butter(4, 0.0045 / (fs/2));
bp = @(x) filtfilt(b1, a1, x) - filtfilt(b2, a2, x);
u = bp(toco);                                % input:  contraction part of the toco
f = bp(fhr);                                 % output: same band of the FHR
figure(2); clf;
subplot(2,1,1); plot(t, u); title('u: band-passed toco');
subplot(2,1,2); plot(t, f); title('f: band-passed FHR'); xlabel('minutes before birth');

%% 4. U: each row is the recent contraction history at one moment
% Row n = [u(n-d), u(n-d-1), ..., u(n-d-M+1)]   (d = delay, M = 60 samples = 120 s)
M = 60;
lagmat = @(d) buildU(u, d, M);
U = lagmat(0);
size(U)                                      % N x 60
U(1000, 1:5)                                 % sample 1000: the 5 most recent toco values

%% 5. Fit h for every delay from -60 s to +60 s and keep the best
fit = @(U, y) (U'*U + 0.05*trace(U'*U)/M*eye(M)) \ (U'*y);   % least squares, lightly regularized
D = -30:30;   vaf = zeros(size(D));
for i = 1:numel(D)
    Ud = lagmat(D(i));   hd = fit(Ud, f);
    vaf(i) = 100 * (1 - var(f - Ud*hd) / var(f));
end
[~, ib] = max(vaf);   d = D(ib);
U = lagmat(d);   h = fit(U, f);
G = sum(h)                                   % gain: bpm per unit of sustained toco
delay_s = d / fs
VAF = vaf(ib)                                % % of band-passed FHR explained
figure(3); clf;
subplot(1,2,1); plot(D/fs, vaf); xlabel('delay d (s)'); ylabel('VAF %'); title('Delay search');
subplot(1,2,2); plot(delay_s + (0:M-1)/fs, h, 'LineWidth', 1.5); yline(0);
xlabel('seconds after pressure'); ylabel('bpm per toco unit'); title(sprintf('h, G = %.2f', G));

%% 6. Contraction-driven part of the FHR
response = U * h;                            % weighted moving sum of the toco
figure(4); clf; plot(t, f, 'Color', [.6 .6 .6]); hold on; plot(t, response, 'r', 'LineWidth', 1.2);
legend('band-passed FHR', 'response = U*h'); xlabel('minutes before birth'); ylabel('bpm');

%% 7. Change the gain: scale only that part, then look at the full FHR
k = 2;                                       % try 0, 0.5, 2, 3, -1
fhr_edit = fhr + (k - 1) * response;         % baseline and variability untouched
figure(5); clf; plot(t, fhr, 'Color', [.6 .6 .6]); hold on; plot(t, fhr_edit, 'k');
legend('recorded', sprintf('gain x %g', k)); xlabel('minutes before birth'); ylabel('FHR (bpm)');
xlim([-30 0]);

%% 8. How the gain moves the other features (last 30 min)
ks = [0 0.5 1 2 3];
T  = table();
for kk = ks
    T = [T; struct2table(ctgFeatures(fhr + (kk - 1) * response, fs, kk))]; %#ok<AGROW>
end
disp(T)

%% ---------------------------------------------------------------- helpers
function U = buildU(u, d, M)
    N = numel(u);   U = zeros(N, M);
    for j = 1:M
        idx = (1:N)' - d - (j - 1);   ok = idx >= 1 & idx <= N;
        U(ok, j) = u(idx(ok));
    end
end

function s = ctgFeatures(x, fs, k)
    % Simple last-30-min features: baseline = 10-min moving median; a deceleration is
    % >= 15 s more than 15 bpm below baseline.
    x   = x(end - 30*60*fs + 1:end);
    bl  = movmedian(x, 10*60*fs);
    dev = x - bl;
    in  = dev < -15;
    edges = diff([0; in; 0]);   st = find(edges == 1);   en = find(edges == -1) - 1;
    keep = (en - st + 1) / fs >= 15;   st = st(keep);   en = en(keep);
    depth = zeros(numel(st), 1);   area = 0;
    for i = 1:numel(st)
        seg = dev(st(i):en(i));
        depth(i) = -min(seg);
        area = area - sum(seg) / fs;
    end
    s.gain_k            = k;
    s.baseline_bpm      = median(bl);
    s.n_decels          = numel(st);
    s.mean_depth_bpm    = mean(depth);
    s.total_area_bpm_s  = area;
    s.time_in_decels_s  = sum(en - st + 1) / fs;
end
