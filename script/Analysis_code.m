% 2026.5.24 edited by Jin Wang.
clear all;
clc;

%% 3.1 Highly alignment in the static representation of words between humans and LLMs.
% 01 Attribute-wise similarities
load("exp_420_level_rating_matrix.mat")    % input the path, load the data.
model = {'chatgpt', 'deepseek', 'gemini', 'qwen'};
model_data = {chatgpt_mean_exp, deepseek_mean_exp, gemini_mean_exp, qwen_mean_exp};

avg_4 = @(x) squeeze(mean(reshape(x, 4, size(x,1)/4, size(x,2)), 1));    % averaged the sentence data(420) into word data(105)
human = avg_4(sub_mean_exp);
corr_results = struct();

fprintf('=== Mean attribute-wise similarity ===\n');
for i = 1:length(model_data)
    current_model = avg_4(model_data{i}); 
    res = zeros(53, 2);    
    for col = 1:53
        [res(col,1), res(col,2)] = corr(current_model(:, col), human(:, col), 'Type', 'Spearman');  % correlation between 4 models and human
    end
    corr_results.(model{i}) = res;   
    mean_r = mean(res(:, 1), 'omitnan'); 
    fprintf('Agent: %-12s | Mean Spearman Rho = %.4f\n', model{i}, mean_r);
end




% 01.1 Mann–Whitney U
idx_sensory = 1:36;          
idx_social = 37:53;      
n1 = length(idx_sensory);
n2 = length(idx_social);

result_strs = cell(1, 4);
fprintf('=== Within LLMs: Perceptual-Motor vs Social-affective (Mann-Whitney U) ===\n');
for i = 1:length(model)        
    rho = corr_results.(model{i})(:, 1); 
    data_sensory = rho(idx_sensory);
    data_social = rho(idx_social);   
    [p, h, stats] = ranksum(data_social, data_sensory); 
    
    R1 = stats.ranksum;
    U1 = R1 - (n2 * (n2 + 1)) / 2; 
    U2 = (n1 * n2) - U1; 
    U_stat = min(U1, U2); 
    r_biserial = 1 - (2 * U1) / (n1 * n2);

    result_strs{i} = sprintf('%s: U(%d, %d) = %d, %.3f, rank-biserial correlation = %.2f',...
        model{i}, n2, n1, round(U1), p, abs(r_biserial));
end
fprintf('%s\n', strjoin(result_strs, ';\n'));


% 01.2 Paied t-test
fprintf('=== Across LLMs: Comparison of Agent (Paired T-test with FDR Correction) ===\n');
n = 53; 
num_pairs = 6; % C(4,2) = 6 comparsions

Comparison = strings(num_pairs, 1);
t_stat = zeros(num_pairs, 1);
df = zeros(num_pairs, 1);
p_raw = zeros(num_pairs, 1);
cohens_d = zeros(num_pairs, 1);

idx = 1;
for i = 1:3
    for j = (i+1):4
        a = rho_all(:, i);
        b = rho_all(:, j);
        [~, p_val, ~, stats] = ttest(a, b);     
        Comparison(idx) = sprintf('%s vs %s', model{i}, model{j});
        t_stat(idx) = stats.tstat;
        df(idx) = stats.df;
        p_raw(idx) = p_val;
        cohens_d(idx) = abs(stats.tstat) / sqrt(n);     % effect size
        idx = idx + 1;
    end
end


% Benjamini-Hochberg FDR correction
[p_sorted, sort_idx] = sort(p_raw);
p_fdr_sorted = p_sorted .* (num_pairs ./ (1:num_pairs)');

for k = num_pairs-1:-1:1
    p_fdr_sorted(k) = min(p_fdr_sorted(k), p_fdr_sorted(k+1));
end
p_fdr_sorted = min(p_fdr_sorted, 1);  
p_fdr = zeros(num_pairs, 1);
p_fdr(sort_idx) = p_fdr_sorted;


results_table = table(Comparison, t_stat, df, p_raw, p_fdr, cohens_d, ...
    'VariableNames', {'Comparison', 't_stat', 'df', 'P_uncorrected', 'P_FDR_corrected', 'Cohens_d'});
disp(results_table);
% ========================================== %





% 02 Similarity in salience profiles
load("exp_420_level_rating_matrix.mat")        % input the path, load the data.
agents = {'human', 'chatgpt', 'deepseek', 'gemini', 'qwen'};
agent_data = {sub_mean_exp, chatgpt_mean_exp, deepseek_mean_exp, gemini_mean_exp, qwen_mean_exp};

profiles = zeros(53,5);
for i = 1: length(agents)
    current_data = agent_data{i};
    profiles(:,i) = mean(current_data,1)';   
end
profiles_table = array2table(profiles,'VariableNames', agents,'RowNames', arrayfun(@(x) sprintf('item%02d', x), 1:53, 'UniformOutput', false));







%% 3.2 Contextual Modulation of Embodied Representations in LLMs and Human 

% 01 Contextual effects across all words
% 01.1 intra-sense distance
load('exp_420_level_rating_matrix.mat');    % input the path, load the data.
agents = {'human', 'chatgpt', 'deepseek', 'gemini', 'qwen'};
agent_data = {sub_mean_exp, chatgpt_mean_exp, deepseek_mean_exp, gemini_mean_exp, qwen_mean_exp};

samesense = struct();
intra_D = struct();
for i = 1:length(agents)
    agent_name = agents{i};
    data = agent_data{i};
    temp_result = NaN(105, 2);       % 2 inter-sense pairs
        for j = 1:4:420
        row_idx = 1 + (j-1)/4;   
        temp_result(row_idx, 1) = corr(data(j,:)', data(j+1,:)', 'Type', 'Spearman');    % sense1_exp1 & sense1_exp2
        temp_result(row_idx, 2) = corr(data(j+2,:)', data(j+3,:)', 'Type', 'Spearman');  % sense2_exp1 & sense2_exp2
    end
    samesense.(agent_name) = temp_result;
end


fprintf('=== Intra-sense distance ===\n');
for i = 1:length(agents)
    agent_name = agents{i};
    intra_D.(agent_name) = 1 - mean(samesense.(agent_name), 2, 'omitnan');   % the 6th row of Qwen has NaN
    mean_val = mean(intra_D.(agent_name), 'omitnan');
    sd_val = std(intra_D.(agent_name), 'omitnan');
    display_name = [upper(agent_name(1)), agent_name(2:end)]; 
    fprintf('%s: Mean = %.4f, SD = %.4f\n', display_name, mean_val, sd_val);
end



% 01.2 inter-sense distance
diffsense = struct();
inter_D = struct();
for i = 1:length(agents)
    agent_name = agents{i};
    data = agent_data{i};
    temp_result = NaN(105, 4);     % 4 inter-sense pairs
        for j = 1:4:420
        row_idx = 1 + (j-1)/4;    
        temp_result(row_idx, 1) = corr(data(j,:)', data(j+2,:)', 'Type', 'Spearman');      %  sense1_exp1 & sense2_exp1
        temp_result(row_idx, 2) = corr(data(j,:)', data(j+3,:)', 'Type', 'Spearman');      %  sense1_exp1 & sense2_exp2
        temp_result(row_idx, 3) = corr(data(j+1,:)', data(j+2,:)', 'Type', 'Spearman');    %  sense1_exp2 & sense2_exp1
        temp_result(row_idx, 4) = corr(data(j+1,:)', data(j+3,:)', 'Type', 'Spearman');    %  sense1_exp2 & sense2_exp2
    end
    diffsense.(agent_name) = temp_result;
end


fprintf('=== Inter-sense distance ===\n');
for i = 1:length(agents)
    agent_name = agents{i};
    inter_D.(agent_name) = 1 - mean(diffsense.(agent_name), 2, 'omitnan');   % the 6th row of Qwen has NaN
    mean_val = mean(inter_D.(agent_name), 'omitnan');
    sd_val = std(inter_D.(agent_name), 'omitnan');
    display_name = [upper(agent_name(1)), agent_name(2:end)]; 
    fprintf('%s: Mean = %.4f, SD = %.4f\n', display_name, mean_val, sd_val);
end



% 01.3 Paired t-test between inter- and intra-sense distance 
agents = {'human', 'chatgpt', 'deepseek', 'gemini', 'qwen'};
fprintf('=== Intra-distance vs Inter-sense distance ===\n');
fprintf('%-10s %-10s %-12s %-10s %-10s %-15s\n', 'Agent', 'Mean_Diff', 't_stat', 'df', 'Cohen_d', 'P_value');

for i = 1:length(agents)
    agent = agents{i};   
    d_intra = intra_D.(agent);   % intra-sense distance as baseline
    d_inter = inter_D.(agent);   % inter-sense distance
    
    [~, p, ~, stats] = ttest(d_intra, d_inter);
    diff_vec = d_intra - d_inter;
    mean_diff = mean(diff_vec, 'omitnan');   
    cohen_d = mean_diff / std(diff_vec, 'omitnan');
    display_name = [upper(agent(1)), agent(2:end)];
    fprintf('%-10s %-10.4f %-12.4f %-10d %-10.4f %-15.4f\n', ...
    display_name, mean_diff, stats.tstat, stats.df, cohen_d, p);
end




% 01.4 One-way ANOVA of inter-sense distance
num_agents = 5;
all_data = [];
group = [];
for i = 1: length(agents)
    val = inter_D.(agents{i});
    all_data = [all_data; val];
    group = [group; repmat(i, length(val), 1)];
end

[p_anova, tbl, ~] = anova1(all_data, group, 'off');    % anova
eta_squared = tbl{2,2} / tbl{4,2};
fprintf('=== One-way ANOVA of inter-sense distance ===\n');
fprintf('F(%d, %d) = %.4f, P = %.4f, eta_squared = %.4f\n\n', tbl{2,3}, tbl{3,3}, tbl{2,5}, p_anova, eta_squared);


% comparison
num_pairs = nchoosek(num_agents, 2);   % C(5,2) = 10 comparisons
Comparison = strings(num_pairs, 1);
Mean_Diff = zeros(num_pairs, 1);
CI_Lower = zeros(num_pairs, 1);
CI_Upper = zeros(num_pairs, 1);
Cohens_d = zeros(num_pairs, 1);
P_raw = zeros(num_pairs, 1);

idx = 1;
for i = 1:(num_agents-1)
    for j = (i+1):num_agents
        a = inter_D.(agents{i});
        b = inter_D.(agents{j});
        
        [~, p_val, ci, stats_t] = ttest2(a, b);    % independent t-test
        
        n_a = length(a); n_b = length(b);
        sd_pool = sqrt(((n_a - 1)*var(a) + (n_b - 1)*var(b)) / (n_a + n_b - 2));
        d_val = (mean(a) - mean(b)) / sd_pool;
        
        
        Comparison(idx) = sprintf('%s vs %s', agents{i}, agents{j});
        Mean_Diff(idx) = mean(a) - mean(b);
        CI_Lower(idx) = ci(1);
        CI_Upper(idx) = ci(2);
        Cohens_d(idx) = abs(d_val); 
        P_raw(idx) = p_val;
        
        idx = idx + 1;
    end
end


% Benjamini-Hochberg FDR correction
[p_sorted, sort_idx] = sort(P_raw);     
p_fdr_sorted = p_sorted .* (num_pairs ./ (1:num_pairs)');

for k = num_pairs-1:-1:1
    p_fdr_sorted(k) = min(p_fdr_sorted(k), p_fdr_sorted(k+1));
end

p_fdr_sorted = min(p_fdr_sorted, 1); 
P_FDR = zeros(num_pairs, 1);
P_FDR(sort_idx) = p_fdr_sorted;

results_table = table(Comparison, Mean_Diff, CI_Lower, CI_Upper, Cohens_d, P_raw, P_FDR);
disp('=== Post hoc pairwise comparison (FDR-corrrected) ===');
disp(results_table);
% =================================







% 02 Contextual effects across all words
num_agents = 5;
all_data = [];
group_agent = {};
group_category = {};
cat_labels = [repmat({'Adjective'}, 35, 1); ...
              repmat({'Noun'}, 35, 1); ...
              repmat({'Verb'}, 35, 1)];

for i = 1:num_agents
    val = inter_D.(agents{i});
    all_data = [all_data; val];
    group_agent = [group_agent; repmat(agents(i), 105, 1)];
    group_category = [group_category; cat_labels];
end


% 02.1 Two-way ANOVA
[p_anova, tbl, ~] = anovan(all_data, {group_agent, group_category}, ...
    'model', 'interaction', 'varnames', {'Agent', 'Category'}, 'display', 'off');
fprintf('=== 双因素 ANOVA (Agent & Category) ===\n');
anova_table = cell2table(tbl(2:end, [1,2,3,5,6,7]), 'VariableNames', {'Source', 'SS', 'df', 'MS', 'F', 'P_value'});
anova_table.Eta_squared = anova_table.SS ./ anova_table.SS(end);
disp(anova_table);

% Post hoc comparisons for AGENTS (t-test + FDR)
num_pairs = nchoosek(num_agents, 2);  % 10 comparisons
Comparison = strings(num_pairs, 1);
Mean_Diff = zeros(num_pairs, 1);
CI_Lower = zeros(num_pairs, 1);
CI_Upper = zeros(num_pairs, 1);
Cohens_d = zeros(num_pairs, 1);
P_raw = zeros(num_pairs, 1);

idx = 1;
for i = 1:(num_agents-1)
    for j = (i+1):num_agents
        a = inter_D.(agents{i});
        b = inter_D.(agents{j});
        
        [~, p_val, ci, stats_t] = ttest2(a, b);   
        n_a = length(a); n_b = length(b);
        sd_pool = sqrt(((n_a - 1)*var(a) + (n_b - 1)*var(b)) / (n_a + n_b - 2));
        
        Comparison(idx) = sprintf('%s vs %s', agents{i}, agents{j});
        Mean_Diff(idx) = mean(a) - mean(b);
        CI_Lower(idx) = ci(1);
        CI_Upper(idx) = ci(2);
        Cohens_d(idx) = abs((mean(a) - mean(b)) / sd_pool);
        P_raw(idx) = p_val;
        idx = idx + 1;
    end
end

[p_sorted, sort_idx] = sort(P_raw);
p_fdr_sorted = p_sorted .* (num_pairs ./ (1:num_pairs)');
for k = num_pairs-1:-1:1
    p_fdr_sorted(k) = min(p_fdr_sorted(k), p_fdr_sorted(k+1));
end
P_FDR = zeros(num_pairs, 1);
P_FDR(sort_idx) = min(p_fdr_sorted, 1);

results_agent = table(Comparison, Mean_Diff, CI_Lower, CI_Upper, Cohens_d, P_raw, P_FDR);
fprintf('\n=== Post hoc comparisons of agents (t-test + FDR correction) ===\n');
disp(results_agent);



% Post hoc comparisons for LEXICAL CATEGORIES (t-test + FDR)
cats = {'Adjective', 'Noun', 'Verb'};
num_cat_pairs = 3;  % C(3,2) = 3 comparisons

Comp_cat = strings(num_cat_pairs, 1);
MD_cat = zeros(num_cat_pairs, 1);
CIL_cat = zeros(num_cat_pairs, 1);
CIU_cat = zeros(num_cat_pairs, 1);
D_cat = zeros(num_cat_pairs, 1);
P_cat = zeros(num_cat_pairs, 1);

idx = 1;
for i = 1:2
    for j = (i+1):3
        a = all_data(strcmp(group_category, cats{i}));
        b = all_data(strcmp(group_category, cats{j}));
        
        [~, p_val, ci, stats_t] = ttest2(a, b);
        n_a = length(a); n_b = length(b);
        sd_pool = sqrt(((n_a - 1)*var(a) + (n_b - 1)*var(b)) / (n_a + n_b - 2));
        
        Comp_cat(idx) = sprintf('%s vs %s', cats{i}, cats{j});
        MD_cat(idx) = mean(a) - mean(b);
        CIL_cat(idx) = ci(1);
        CIU_cat(idx) = ci(2);
        D_cat(idx) = abs((mean(a) - mean(b)) / sd_pool);
        P_cat(idx) = p_val;
        idx = idx + 1;
    end
end

[p_sorted_c, sort_idx_c] = sort(P_cat);
p_fdr_sorted_c = p_sorted_c .* (num_cat_pairs ./ (1:num_cat_pairs)');
for k = num_cat_pairs-1:-1:1
    p_fdr_sorted_c(k) = min(p_fdr_sorted_c(k), p_fdr_sorted_c(k+1));
end
P_FDR_cat = zeros(num_cat_pairs, 1);
P_FDR_cat(sort_idx_c) = min(p_fdr_sorted_c, 1);

results_category = table(Comp_cat, MD_cat, CIL_cat, CIU_cat, D_cat, P_cat, P_FDR_cat, ...
    'VariableNames', {'Comparison', 'Mean_Diff', 'CI_Lower', 'CI_Upper', 'Cohens_d', 'P_raw', 'P_FDR'});
fprintf('\n=== Post hoc comparisons of lexical categories (t-test + FDR correction) ===\n');
disp(results_category);







%% 3.3  Cross-dimensional coupling of Embodied representation in LLMs and Humans 

% 01 Cross-dimensional coupling across all words
% 01.1 clear the data
load("exp_420_level_rating_matrix.mat");      % input the path, load the data.
cols_to_remove = [3, 7, 9, 14, 15, 32, 36, 45];               % Main text analysis: remove 8 negative-keyed attributes
% cols_to_remove_negative = [2, 6, 8, 13, 16, 31, 35, 44];    % Supplementary analysis: remove 8 positive-keyed attributes

agents = {'sub', 'chatgpt', 'deepseek', 'gemini', 'qwen'};
n_agents = 5;
for m = 1:n_agents
    original_data = eval([agents{m} '_mean_exp']);                   %  Main text analysis
    original_data(:, cols_to_remove) = [];         % remove columns
    eval([agents{m} '_clear_mean = original_data;']);   
%     original_data = eval(['negative_' agents{m} '_mean_exp']);     %　Supplementary analysis
%     original_data(:, cols_to_remove_negative) = [];        
%     eval(['negative_' agents{m} '_clear_mean = original_data;']);   
end



% 01.2 divide the data into two dimensions
sensory = [1:29];
social = [30:45];
for m = 1:n_agents
    original_data =  eval([agents{m} '_clear_mean']);                                       %  Main text analysis
    eval([agents{m} '_clear_sensory = mean(original_data(:,sensory),2);']);                
    eval([agents{m} '_clear_social = mean(original_data(:,social),2);']);
%     original_data =  eval(['negative_' agents{m} '_clear_mean']);                         %　Supplementary analysis
%     eval(['negative_' agents{m} '_clear_sensory = mean(original_data(:,sensory),2);']);   
%     eval(['negative_' agents{m} '_clear_social = mean(original_data(:,social),2);']);
end



% 01.3 calculate the coupling
for m = 1:n_agents
    sensory_data = eval([agents{m} '_clear_sensory']);                   %  Main text analysis
    social_data = eval([agents{m} '_clear_social']);
    [r(m), p(m)] = corr(sensory_data, social_data, "type", "Spearman");
    fprintf('(Positive) %s: r = %.4f, p = %f\n', agents{m}, r(m), p(m));      
%     sensory_data = eval(['negative_' agents{m} '_clear_sensory']);     %　Supplementary analysis
%     social_data = eval(['negative_' agents{m} '_clear_social']);
%     [r(m), p(m)] = corr(sensory_data, social_data, "type", "Spearman");
%     fprintf('(Negative) %s: r = %.4f, p = %f\n', agents{m}, r(m), p(m));  　
end



% 01.4 Fisher z test of coupling: comparisons across agents.
agents = {'sub', 'chatgpt', 'deepseek', 'gemini', 'qwen'};
cp = r;      % the couplings of 5 agents
n = 420;     % the original data used to calculate the coupling has 420 rows
z = 0.5 * log((1 + cp) ./ (1 - cp));
results = table();
row_idx = 0; 

for i = 1:(n_agents-1)
    for j = (i+1):n_agents
        row_idx = row_idx + 1; 
        z_diff = z(i) - z(j);
        z_se = sqrt(2/(n-3));   
        z_stat = z_diff / z_se;
        p_val = 2 * (1 - normcdf(abs(z_stat)));
        results.Comparison{row_idx, 1} = [agents{i} ' vs ' agents{j}];
        results.r_agent1(row_idx, 1)   = cp(i);
        results.r_agent2(row_idx, 1)   = cp(j);
        results.z_stat(row_idx, 1)     = z_stat;
        results.p_value(row_idx, 1)    = p_val;
    end
end
disp(results);
% ========================================








% 02 Cross-dimensional coupling varied by lexical category
% 02.1 divide the data by lexical category
agents = {'sub', 'chatgpt', 'deepseek', 'gemini', 'qwen'};
n_agents = 5;
adj = [1:140]; noun = [141:280]; verb = [281:420];
for m = 1:n_agents
    original_data = eval([agents{m} '_clear_mean']);
    eval([agents{m} '_clear_mean_adj = original_data(adj, :);']);     % For supplementary analysis, just add a 'negative_' before the agents{m}
    eval([agents{m} '_clear_mean_noun = original_data(noun, :);']);
    eval([agents{m} '_clear_mean_verb = original_data(verb, :);']);
end


% 02.2 divide the prepared data into two dimensions
for m = 1:n_agents
    original_data_adj = eval([agents{m} '_clear_mean_adj']);
    original_data_noun = eval([agents{m} '_clear_mean_noun']);
    original_data_verb = eval([agents{m} '_clear_mean_verb']);
    eval([agents{m} '_clear_adj_sensory = mean(original_data_adj(:,sensory),2);']);
    eval([agents{m} '_clear_adj_social = mean(original_data_adj(:,social),2);']);
    eval([agents{m} '_clear_noun_sensory = mean(original_data_noun(:,sensory),2);']);
    eval([agents{m} '_clear_noun_social = mean(original_data_noun(:,social),2);']);
    eval([agents{m} '_clear_verb_sensory = mean(original_data_verb(:,sensory),2);']);
    eval([agents{m} '_clear_verb_social = mean(original_data_verb(:,social),2);']);
end



% 02.3 calculate the coupling
lexical = {'adj','noun','verb'};
n_agents = 5;
n_lexcial = 3;
for m = 1:n_agents
    for i = 1:n_lexcial
        sensory_data = eval([agents{m} '_clear_' lexical{i} '_sensory']);
        social_data = eval([agents{m} '_clear_' lexical{i} '_social']);
        [r(m*3-3+i), p(m*3-3+i)] = corr(sensory_data, social_data, "type", "Spearman");      
        fprintf('%s(%s): r = %.4f, p = %f\n', agents{m}, lexical{i}, r(m*3-3+i), p(m*3-3+i))    % the index
    end
end
% ========================================











%% Supplementary analysis
% 01 Representational couplings across four experiential modalities: Comparison of within- and cross-pairs coupling
load("exp_420_level_rating_matrix.mat");        % input the path, load the data.
agents = {'Human', 'ChatGPT', 'DeepSeek', 'Gemini', 'Qwen'};
data_cells = {sub_mean_exp, chatgpt_mean_exp, deepseek_mean_exp, gemini_mean_exp, qwen_mean_exp};

cat = [ones(1, 36), 2*ones(1, 17)];     % Perceptual-motor: 1-36; Social-affective: 37-53
[C1, C2] = meshgrid(cat, cat);
idx_upper = triu(true(53), 1);        
within_mask = (C1 == C2) & idx_upper;   % within-dimension pairs =  766 
cross_mask  = (C1 ~= C2) & idx_upper;   % cross-dimension pairs =  612 (C36*C17)
within_r = cell(1, 5);
cross_r  = cell(1, 5);

for i = 1:5
    R = corr(data_cells{i}, 'Type', 'Spearman');  % calculate 53x53 dimensions coupling
    within_r{i} = R(within_mask);                 % within-dimension coupling
    cross_r{i}  = R(cross_mask);                  % cross-dimension coupling
end

% Mann-Whitney U Test (Models vs Human)
fprintf('\n================================= Mann-Whitney U Test (Model vs Human) =================================\n');
fprintf('%-10s | %-32s | %-32s\n', 'LLM Agent', 'Within Pairs (Agent - Human)', 'Cross Pairs (Agent - Human)');
fprintf('%-10s | %-8s %-11s %-10s | %-8s %-11s %-10s\n', '', 'r_diff', 'p-value', 'z-stat', 'r_diff', 'p-value', 'z-stat');
fprintf('--------------------------------------------------------------------------------------------------\n');

% Calculate the averaged within- cross-dimensional coupling of Agents
mean_human_w = mean(within_r{1}, 'omitnan');
mean_human_c = mean(cross_r{1}, 'omitnan');

for i = 2:5
    r_diff_w = mean(within_r{i}, 'omitnan') - mean_human_w;
    r_diff_c = mean(cross_r{i}, 'omitnan') - mean_human_c;
    [p_w, ~, stats_w] = ranksum(within_r{1}, within_r{i});   % within-dimension coupling
    [p_c, ~, stats_c] = ranksum(cross_r{1}, cross_r{i});     % cross-dimension coupling
    fprintf('%-10s | %-8.4f %-11.4e %-10.2f | %-8.4f %-11.4e %-10.2f\n', ...
            agents{i}, r_diff_w, p_w, stats_w.zval, r_diff_c, p_c, stats_c.zval);
end
%=======================






% 02  Modalities coupling 
models = {'sub', 'chatgpt', 'deepseek', 'gemini', 'qwen'};
n_models = length(models);

pair_names = {
    'Motor_vs_Perceptual_Rho', 'Motor_vs_Perceptual_p',...
    'Motor_vs_Social_Rho', 'Motor_vs_Social_p',...
    'Motor_vs_Affective_Rho', 'Motor_vs_Affective_p',...
    'Perceptual_vs_Social_Rho', 'Perceptual_vs_Social_p',...
    'Perceptual_vs_Affective_Rho', 'Perceptual_vs_Affective_p',...
    'Social_vs_Affective_Rho', 'Social_vs_Affective_p'
};

results = zeros(n_models, length(pair_names));

for m = 1:n_models
    % 获取数据
    perceptual = eval([models{m} '_clear4_sensory']);     % remove the negatively keyed attributes
    motor = eval([models{m} '_clear4_motor']);
    social = eval([models{m} '_clear4_social']);
    affective = eval([models{m} '_clear4_emotion']);

    variables = [perceptual, motor, social, affective];

    [R, P] = corr(variables, 'Type', 'Spearman', 'Rows', 'pairwise');

    rho_values = [
        R(2,1),  % Motor_vs_Perceptual
        R(3,1),  % Motor_vs_Social
        R(4,1),  % Motor_vs_Affective
        R(3,2),  % Perceptual_vs_Social
        R(4,2),  % Perceptual_vs_Affective
        R(4,3)   % Social_vs_Affective
    ];
    
    p_values = [
        P(2,1),  % Motor_vs_Perceptual
        P(3,1),  % Motor_vs_Social
        P(4,1),  % Motor_vs_Affective
        P(3,2),  % Perceptual_vs_Social
        P(4,2),  % Perceptual_vs_Affective
        P(4,3)   % Social_vs_Affective
    ];
    
    for p = 1:6
        results(m, (p-1)*2+1) = rho_values(p);         % saved the rho and p value
        results(m, (p-1)*2+2) = p_values(p);
    end
end





% The difference of coupling (fisher z-test)
agents = {'Human', 'ChatGPT', 'DeepSeek', 'Gemini', 'Qwen'};
target_cols = [1, 3, 5, 7, 9, 11];   % column 1/3/5/7/9/11 is the correlation data.
n = 420;  % sample size
pair_names = {
    'Motor_vs_Perceptual', ...
    'Motor_vs_Social', ...
    'Motor_vs_Affective', ...
    'Perceptual_vs_Social', ...
    'Perceptual_vs_Affective', ...
    'Social_vs_Affective'
};


fprintf('\n================== Fisher Z-test (Agent vs Human) ==================\n');
fprintf('%-10s | %-25s | %-8s | %-10s | %-10s\n', 'Agent', 'Dimension Pair', 'r_Agent', 'r_diff', 'p-value');
fprintf('--------------------------------------------------------------------\n');

for i = 2:5   % Four models
    agent_name = agents{i};
    for j = 1:length(target_cols)    % 6 modalities pairs
        col_idx = target_cols(j);
        pair_name = pair_names{j};

        r_human = results(1, col_idx);
        r_agent = results(i, col_idx);
        r_diff = r_agent - r_human;     % (r_diff = Agent - Human)  model-human
        
        z_human = 0.5 * log((1 + r_human) / (1 - r_human));             % Fisher Z 
        z_agent = 0.5 * log((1 + r_agent) / (1 - r_agent));
        z_se = sqrt(2 / (n - 3)); 
        z_stat = (z_agent - z_human) / z_se;
        p_val = 2 * (1 - normcdf(abs(z_stat)));

        fprintf('%-10s | %-25s | %-8.4f | %-10.4f | %-10.5f\n', ...
            agent_name, pair_name, r_agent, r_diff, p_val);
    end
    fprintf('--------------------------------------------------------------------\n');
end
% ============================


% Congrats. We made it.




