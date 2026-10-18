%%%fileread

% input paths (set knowledge_csv / expdir before running this script to change them)
if ~exist('knowledge_csv', 'var')
    knowledge_csv = '../knowledge/knowledge.csv';
end
if ~exist('expdir', 'var')
    expdir = '../experiment/';
end
KB = csvread(knowledge_csv);

h = expdir;
fid = fopen(strcat(h, 'timestamp.list'));
tline = fgetl(fid);

session = cell(2, 3);
semantic_attribute_seq = cell(2, 3);
semantic_attributes = cell(2, 3);
count = 0;
usr = 0;
while(ischar(tline))
    session_id = mod(count, 4) + 1;
    if session_id == 1
        usr = usr + 1;
    end
    if session_id ~= 4
        cont = csvread(strcat(h, tline, '.list'));
        data = csvread(strcat(h, tline, '.csv'));
        seq = LoadGaze(data);
        task = floor(cont(1)/100);
        f = seq(:, 1);
        session{usr, task}.f = f;
        session{usr, task}.dur = seq(:, 2);
        list = cont - task * 100;
        session{usr, task}.list = list;
        session{usr, task}.task = task;
        session{usr, task}.usr = usr;
        session{usr, task}.session_id = session_id; % order of the session for this user (1..3)
        session{usr, task}.totalduration = (data(end, 1) - data(1, 1))/1000000; % s
        semantic_attribute_seq{usr, task} = KB(list(f), :)';
        semantic_attributes{usr,task} = KB(list, :);
        if task == 1
            session{usr, task}.candidate = [1 2];
        elseif task == 2
            session{usr, task}.candidate = [8 11];
        elseif task == 3
            session{usr, task}.candidate = [3 4];
        end
    end
    count = count + 1;
    tline = fgetl(fid);
end
fclose(fid);

Set_selected
save('gazedata.mat', 'session', 'semantic_attribute_seq', 'semantic_attributes', 'KB');