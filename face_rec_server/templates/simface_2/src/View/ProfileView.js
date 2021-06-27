var ProfileView = Backbone.View.extend({
    template: _.template($('#profile-template').html()),

    instagramProfileURL: 'https://www.instagram.com/',
    instagramPostURL   : 'https://www.instagram.com/p/',

    events: {
        'mouseenter': 'hightlightOn',
        'mouseleave': 'hightlightOff'
    },

    index: 0,
    
    initialize: function(argc) {
        this.index = argc.index || 0;
    },

    render: function() {
        this.$el.html(
            this.template({
                score:    Number(this.model.get('score') * 100).toFixed(2),
                username: this.model.get('username'),
                post:     this.model.get('post'),

                profileURL: this.instagramProfileURL + this.model.get('username'),
                postURL   : this.instagramPostURL    + this.model.get('post'),
                index: this.index
            })
        );
        
        this.hightlightOff();

        return this;
    },

    hightlightOn: function() {
        this.$('.card')
            .removeClass('is-shadowless');
    },    

    hightlightOff: function() {
        this.$('.card')
            .addClass('is-shadowless');
    }
});