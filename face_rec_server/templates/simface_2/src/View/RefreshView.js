var RefreshView = Backbone.View.extend({
    template: _.template($('#refresh-template').html()),
    deactiveOpacity:    '0.33',

    events: {
        'mouseenter' : 'active',
        'mouseleave' : 'deactive',
        'click'      : 'refresh'
    },

    initialize: function() {
        Backbone.on('refresh.animationShow', this.animationShow, this);
        Backbone.on('refresh.animationHide', this.animationHide, this);
    },

    render: function() {
        this.$el.html(
            this.template()
        );

        this.deactive();
        this.hide();

        return this;
    },

    active: function() {
        this.$el.css('opacity', '1.0');
    },

    deactive: function() {
        this.$el.css('opacity', this.deactiveOpacity);
    },
    
    refresh: function() {
        this.$('.fa-sync').addClass('fa-spin');
        var $this = this;
        Backbone.trigger('flipper.logo&photo');
        Backbone.trigger('flipper.upload&profiles');

        setTimeout(function() {
            $this.animationHide(function() {
                $this.$('.fa-sync').removeClass('fa-spin');
            });
        }, 1000);
    },
});

_.extend(RefreshView.prototype, AnimationMixin);