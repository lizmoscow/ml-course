var AppView = Backbone.View.extend({
    template: _.template($('#app-template').html()),

    initialize: function(argc) {
        this.version = argc.version || 'unknow';

        this.flipperLogoAndPhoto = new FlipperView({
            heads: new LogoView({
                showAnimationClass: 'fadeInDown',
                hideAnimationClass: 'fadeOutUp'
            }),
            tails: new PhotoView({
                showAnimationClass: 'fadeInDown',
                hideAnimationClass: 'fadeOutUp'
            }),
            triggerName: 'flipper.logo&photo',
            template   : '#flipper-logo-and-photo-template'
        });

        this.flipperUploadAndProfiles = new FlipperView({
            heads: new UploadView({
                model: this.model.get('uploadModel'),
                showAnimationClass: 'fadeIn',
                hideAnimationClass: 'fadeOut'
            }),
            tails: new ProfileListView({
                profiles: this.model.get('profileList'),
                showAnimationClass: 'fadeInUpBig',
                hideAnimationClass: 'fadeOutDownBig'
            }),
            triggerName: 'flipper.upload&profiles',
            template   : '#flipper-upload-and-profiles-template',
            initializeAnimation: false
        })

        this.refresh = new RefreshView();

        Backbone.on('app.reinitialize', this.reinitialize, this);
    },

    reinitialize: function() {
        this.logoView.swap();
    },

    render: function() {

        this.$el.html(
            this.template()
        );

        this.$('#version').html(this.version);

        this.flipperLogoAndPhoto.setElement(this.$('#flipper-logo-and-photo'));
        this.flipperUploadAndProfiles.setElement(this.$('#flipper-upload-and-profiles'));

        this.refresh.setElement(this.$('#refresh'));

        this.flipperLogoAndPhoto.render();
        this.flipperUploadAndProfiles.render();

        this.refresh.render();

        return this;
    }
});