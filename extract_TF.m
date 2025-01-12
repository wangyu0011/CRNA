%%
load('matlab_phase.mat')
PATH1='/.../Ultimatum_Game/ERP_data2/';
PATH2='/.../Ultimatum_Game/TF_data/';
cd(PATH1);
list=dir('proposer*.set');
for i=1:length(list)
    EEG = pop_loadset('filename',list(i).name,'filepath',PATH1);
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    for k2=1:EEG.trials
        DATA=zeros(128,42,200);
        for t=1:128
            data=EEG.data(t,:,k2)';
            [alltfX freqs timesout R] = timefreq(data, g.srate, tmioutopt{:}, ...
                    'winsize', 128, 'tlimits', g.tlimits, 'detrend', g.detrend, ...
                    'itctype', g.type, 'wavelet', g.cycles, 'verbose', g.verbose, ...
                    'padratio', 8, 'freqs', g.freqs, 'freqscale', g.freqscale, ...
                    'nfreqs', g.nfreqs, 'timestretch', {g.timeStretchMarks', g.timeStretchRefs}, timefreqopts{:});
            DATA(t,:,:)=alltfX;
        end
        A=list(i).name;
        A1=A(1:end-4);
        save([PATH2,A1,'_',num2str(k2),'.mat'],'DATA');
    end
    STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
end



