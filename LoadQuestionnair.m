%%% load questionnair

withASP = [[1 1 1 2]; [3 2 2 4]; [1 1 1 4]; [2 2 1 3]; [2 4 2 4]; [1 1 1 5]; [2 1 1 5]; [1 1 1 4]; [2 2 2 3]];
withoutASP = [[2 2 2 2]; [3 2 3 3]; [1 1 1 4]; [2 1 2 2]; [1 1 1 2]; [1 2 1 5]; [1 1 1 5]; [3 2 2 4]];

withASP(:, 1 : 3) = 6 - withASP(:, 1 : 3);
withoutASP(:, 1 : 3) = 6 - withoutASP(:, 1 : 3);

bar([mean(withASP)' mean(withoutASP)'])
set(gca,'xticklabel',{'‘I‘ğ‚Ì–‘«Š´','Šî€‚Ì–¾Šm‚³','ÀÛ‚Ég‚¤‚©','“­‚«‚©‚¯‚ª–ğ‚É—§‚Á‚½‚©'})
legend('with ASP', 'only att')

