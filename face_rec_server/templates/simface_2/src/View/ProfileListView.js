var ProfileListView = Backbone.View.extend({
    template: _.template($('#profilelist-template').html()),
    
    initialize: function(args) {
        Object.assign(this, args);

        Backbone.on('profileList.setProfiles', this.setProfiles, this);
    },

    render: function() {
        
        this.$el.html(
            this.template
        );

        var $this = this;
        this.$el.empty();

        if (this.profiles.length == 0)
            return this;

        Backbone.trigger('flipper.logo&photo.animate', 'slideInUp');

        
        this.profiles.forEach(function(profile, index) {
            var view = new ProfileView({
                model: profile,
                index: index
            });

            var render = view.render();

            $this.$el.append(render.el);
        });


        return this;
    },

    setProfiles: function(profiles) {
        if (profiles) {
            this.profiles.reset();
            this.profiles.add(profiles);
        } 
    }

});

_.extend(ProfileListView.prototype, AnimationMixin);
