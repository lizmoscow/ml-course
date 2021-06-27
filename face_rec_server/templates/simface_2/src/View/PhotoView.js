var PhotoView = Backbone.View.extend({
    template: _.template($('#photo-template').html()),
    base64: null,

    initialize: function(argc) {
        Object.assign(this, argc);
        Backbone.on('photo.setPhoto', this.setPhoto, this);
    },

    setPhoto: function(base64) {
        this.base64 = base64;
    },

    render: function() {
        this.$el.html(
            this.template
        );

        this.$('#photo').attr('src', this.base64);

        return this;
    }
});

_.extend(PhotoView.prototype, AnimationMixin);