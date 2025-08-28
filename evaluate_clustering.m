function [NMI_score, ARI_score, Purity] = evaluate_clustering(true_labels, predicted_labels)
    % Ensure labels are column vectors
    true_labels = true_labels(:);
    predicted_labels = predicted_labels(:);

    % Number of data points
    N = length(true_labels);

    % Unique labels and number of clusters/classes
    classes = unique(true_labels);
    clusters = unique(predicted_labels);

    num_classes = length(classes);
    num_clusters = length(clusters);

    % Contingency Matrix (Confusion Matrix)
    contingency = zeros(num_classes, num_clusters);
    for i = 1:num_classes
        for j = 1:num_clusters
            contingency(i, j) = sum((true_labels == classes(i)) & (predicted_labels == clusters(j)));
        end
    end

    %% --------- NMI Calculation ---------
    % Mutual Information
    MI = 0;
    for i = 1:num_classes
        for j = 1:num_clusters
            if contingency(i, j) > 0
                Nij = contingency(i, j);
                Ni = sum(contingency(i, :));
                Nj = sum(contingency(:, j));
                MI = MI + (Nij / N) * log((Nij * N) / (Ni * Nj));
            end
        end
    end

    % Entropy for true labels
    H_true = 0;
    for i = 1:num_classes
        Ni = sum(contingency(i, :));
        if Ni > 0
            H_true = H_true - (Ni / N) * log(Ni / N);
        end
    end

    % Entropy for predicted labels
    H_pred = 0;
    for j = 1:num_clusters
        Nj = sum(contingency(:, j));
        if Nj > 0
            H_pred = H_pred - (Nj / N) * log(Nj / N);
        end
    end

    % Normalized Mutual Information
    NMI_score = MI / sqrt(H_true * H_pred);

    %% --------- ARI Calculation ---------
    a = sum(sum(contingency .* (contingency - 1))) / 2;
    b = sum(sum(sum(contingency, 2) .* (sum(contingency, 2) - 1))) / 2;
    c = sum(sum(sum(contingency, 1) .* (sum(contingency, 1) - 1))) / 2;
    d = N * (N - 1) / 2;

    expected_index = (b * c) / d;
    max_index = (b + c) / 2;

    ARI_score = (a - expected_index) / (max_index - expected_index);

    %% --------- Purity Calculation ---------
    sum_max = 0;
    for j = 1:num_clusters
        sum_max = sum_max + max(contingency(:, j));
    end
    Purity = sum_max / N;
end
