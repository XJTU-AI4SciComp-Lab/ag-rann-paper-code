function draw_surface_2d(style, X, Y, V)
%DRAW_SURFACE_2D Shared 2-D surface plotting styles used by experiments.

Z = reshape(V, size(X, 1), size(X, 2));

switch string(style)
    case "basic_top"
        % figure
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        % view(-38, 30)
        view(2)
        colorbar

    case "font_square_top"
        % figure
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
        axis square
        % view(-38, 30)
        view(2)
        colorbar

    case "single_square_top"
        figure
        hold on
        surf(X, Y, Z, EdgeColor = 'none');
        axis square
        % view(-38, 30)
        view(2)
        colorbar

    case "unit_box_top"
        figure
        hold on
        surf(X, Y, Z, EdgeColor = 'none');
        view(2)
        axis equal
        axis([0 1 -1 1])
        colorbar

    case "view3_square"
        % figure
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        view(-38, 30)
        colorbar
        axis square

    case "font_square_top_alt"
        % figure
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
        axis square
        view(2)
        % view(3)
        colorbar

    case "plain_square_top"
        % figure
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        % set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
        axis square
        view(2)
        % view(3)
        colorbar

    case "font_square_top_clim"
        % figure
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
        axis square
        view(2)
        clim([-0.1,1.1])
        % view(3)
        colorbar

    case "font_square_3d_clim"
        % figure
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
        axis square
        % view(2)
        view(3)
        clim([-0.1,1.1])
        colorbar

    case "layer_growth"
        figure('Units','centimeter','Position',[10 10 13 10])
        % hold on
        surf(X, Y, Z, EdgeColor = 'none');
        set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
        % view(-38, 30)
        axis square
        view(2)
        colorbar

    otherwise
        error("agrann:draw_surface_2d:UnknownStyle", ...
            "Unknown draw_surface_2d style '%s'.", string(style));
end
end
