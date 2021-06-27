var AnimatableMixin = {

    runAnimation: function(animation, endCallback) {
        var $this = this;
        var animationClass = `animated ${animation}`;
        endCallback = endCallback || function(){};

        this.$el
            .addClass(animationClass)
            .on('animationend', function() {
                $this.$el.removeClass(animationClass);
                $this.removeAnimationEndListener();
                endCallback();
            });
    },

    removeAnimationEndListener: function () {
        this.$el
            .off('animationend');
    }
};