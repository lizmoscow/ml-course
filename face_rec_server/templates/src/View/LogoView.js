var LogoView = Backbone.View.extend({
    template: _.template($('#logo-template').html()),

    initialize: function(argc) {
        Object.assign(this, argc);
    },

    render: function() {
        this.$el.html(
            this.template
        );

        return this;
    }
});

_.extend(LogoView.prototype, AnimationMixin);