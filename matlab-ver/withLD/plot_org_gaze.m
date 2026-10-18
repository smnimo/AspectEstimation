%function target_dur = LoadGaze(data)
%%%fileread

h = '../experiment/';
fid = fopen(strcat(h, 'timestamp.list'));
tline = fgetl(fid);

session = cell(2, 3);
semantic_attribute_seq = cell(2, 3);
semantic_attributes = cell(2, 3);
count = 0;
usr = 0;
while(ischar(tline))
    try 
        cont = csvread(strcat(h, tline, '.list'));
        data = csvread(strcat(h, tline, '.csv'));
        img = imread(strcat(h, tline, '.jpg'));
        figure()
        imshow(img)
        hold on

        timestamp = data(:, 1);
        gaze = data(:, 2:end);

        x = gaze(:, [1,3]);
        y = gaze(:, [2,4]);
        % a missing eye (-1) is filled from the other eye, as in LoadGaze.m
        missing = x == -1;
        x(missing(:, 1), 1) = x(missing(:, 1), 2);
        x(missing(:, 2), 2) = x(missing(:, 2), 1);
        x = mean(x, 2);

        missing = y == -1;
        y(missing(:, 1), 1) = y(missing(:, 1), 2);
        y(missing(:, 2), 2) = y(missing(:, 2), 1);
        y = mean(y, 2);
        
        y(x<0)=[];
        x(x<0)=[];
        x(y<0)=[];
        y(y<0)=[];

        gaze_seq = [x, y];
        gaze_seq = gaze_seq.*repmat([size(img, 2), size(img, 1)], size(gaze_seq, 1), 1);
        %scatter(gaze_seq(:, 1), gaze_seq(:, 2))
        plot(gaze_seq(:, 1), gaze_seq(:, 2))
    end

    tline = fgetl(fid);
end
